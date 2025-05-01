//! Categories over generic objects with power object management.
//!
//! The modules provides the traits and data structures required for basic categorical
//! constructions, supporting generic objects and morphisms, as well as support for power object
//! generation and tracking, and various different build paradigms (supporting both lazy
//! constructions and a builder pattern)

use std::{collections::HashMap, fmt::Debug, hash::Hash};

pub trait Object: Clone + PartialEq + Debug {}

pub trait Morphism {
  type Domain: Object;
  type Codomain: Object;
  fn domain(&self) -> &Self::Domain;
  fn codomain(&self) -> &Self::Codomain;
  fn map(&self, domain: &Self::Domain) -> Self::Codomain;
}

pub fn check_eq_morphisms<A: Object, B: Object>(
  first: &dyn Morphism<Domain = A, Codomain = B>,
  second: &dyn Morphism<Domain = A, Codomain = B>,
) -> bool {
  if first.domain() == second.domain()
    && first.codomain() == second.codomain()
    && first.map(first.domain()) == second.map(second.domain())
  {
    return true;
  }
  false
}

pub fn compose<A: Object, B: Object, C: Object>(
  domain: &A,
  first: &dyn Morphism<Domain = A, Codomain = B>,
  second: &dyn Morphism<Domain = B, Codomain = C>,
) -> C {
  second.map(&first.map(domain))
}

type HomSet<A, B> = Vec<Box<dyn Morphism<Domain = A, Codomain = B>>>;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum PowerObjectType {
  Product(usize, usize),
  Coproduct(usize, usize),
  Exponential(usize, usize),
}

pub trait PowerObjectGenerator<O: Object> {
  fn generate_power_object(
    &self,
    power_type: &PowerObjectType,
    objects: &[O],
  ) -> (O, Vec<Box<dyn Morphism<Domain = O, Codomain = O>>>);
}

/// A `Category` of a single class of object
/// e.g. Vect_k, Hilb_k
pub struct Category<O: Object, P: PowerObjectGenerator<O>> {
  objects:       Vec<O>,
  morphisms:     HashMap<(usize, usize), HomSet<O, O>>,
  power_objects: Option<HashMap<PowerObjectType, usize>>,
  generator:     Option<P>,
}

impl<O: Object, P: PowerObjectGenerator<O>> Category<O, P> {
  /// Helper method to construct a category using the builder pattern
  fn from_builder(
    objects: Vec<O>,
    morphisms: HashMap<(usize, usize), HomSet<O, O>>,
    power_objects: Option<HashMap<PowerObjectType, usize>>,
    generator: Option<P>,
  ) -> Self {
    Self { objects, morphisms, power_objects, generator }
  }

  /// Create a blank category. For the category to support power objects, please provide a
  /// `generator` that is Some.
  pub fn create(generator: Option<P>) -> Self {
    let power_objects = if generator.is_none() { None } else { Some(HashMap::new()) };
    Category { objects: Vec::new(), morphisms: HashMap::new(), power_objects, generator }
  }

  /// Construct a category from a collection of objects with null HomSets
  pub fn from_object_list(objects: &[O], generator: Option<P>) -> Self {
    let power_objects = if generator.is_none() { None } else { Some(HashMap::new()) };
    Category { objects: objects.to_vec(), morphisms: HashMap::new(), power_objects, generator }
  }

  /// Add an object to the current category. Meant for lazy implementations with as needed
  /// constructions.
  pub fn add_object(&mut self, object: O) {
    if !self.objects.contains(&object) {
      self.objects.push(object)
    }
  }

  /// Add a morphism to the current category. Meant for lazy implementations with as needed
  /// constructions.
  pub fn add_morphism(
    &mut self,
    domain: usize,
    codomain: usize,
    map: Box<dyn Morphism<Domain = O, Codomain = O>>,
  ) -> Result<(), String> {
    let actual_domain = map.domain();
    let actual_codomain = map.codomain();

    if self.objects.get(domain) != Some(actual_domain) {
      return Err(format!("Domain index {domain} does not match morphism's domain object"));
    }
    if self.objects.get(codomain) != Some(actual_codomain) {
      return Err(format!("Codomain index {codomain} does not match morphism's codomain object"));
    }

    let key = (domain, codomain);
    if let Some(homset) = self.morphisms.get_mut(&key) {
      let mut insert = true;
      for m in homset.iter() {
        if check_eq_morphisms(m.as_ref(), map.as_ref()) {
          insert = false;
          break;
        }
      }
      if insert {
        homset.push(map);
      }
    } else {
      self.morphisms.insert(key, vec![map]);
    }
    Ok(())
  }

  /// Fetch a power object's ID from the type and constituents
  pub fn fetch_power_object_id(&mut self, power_type: PowerObjectType) -> Result<usize, String> {
    if self.generator.is_none() || self.power_objects.is_none() {
      return Err("Uninitialized power object generator!".to_string());
    }
    if let Some(&idx) = self.power_objects.as_ref().unwrap().get(&power_type) {
      return Ok(idx);
    }

    let (new_obj, new_morphisms) =
      self.generator.as_ref().unwrap().generate_power_object(&power_type, &self.objects);

    self.add_object(new_obj);
    let new_idx = self.objects.len() - 1;
    self.power_objects.as_mut().unwrap().insert(power_type, new_idx);

    for morph in new_morphisms {
      let domain_obj = morph.domain();
      let codomain_obj = morph.codomain();

      let domain_idx = self
        .objects
        .iter()
        .position(|o| o == domain_obj)
        .ok_or("Domain object not found for morphism".to_string())?;
      let codomain_idx = self
        .objects
        .iter()
        .position(|o| o == codomain_obj)
        .ok_or("Codomain object not found for morphism".to_string())?;

      self.add_morphism(domain_idx, codomain_idx, morph)?;
    }

    Ok(new_idx)
  }

  /// Checks whether a morphism is monic or not
  pub fn is_monic(&self, domain: usize, codomain: usize, fn_idx: usize) -> Result<bool, String> {
    let base = &self.morphisms.get(&(domain, codomain)).unwrap()[fn_idx];
    for i in 0..self.objects.len() {
      let homset = self.morphisms.get(&(i, domain));
      if homset.is_none() {
        continue;
      }
      let prior_maps = homset.unwrap();
      for ref_map in prior_maps.iter() {
        let ref_output = ref_map.map(ref_map.domain());
        let ref_chained = base.map(&ref_output);
        for test_map in prior_maps.iter() {
          let test_output = test_map.map(test_map.domain());
          let test_chained = base.map(&test_output);
          if test_chained == ref_chained && test_output != ref_output {
            return Ok(false);
          }
        }
      }
    }
    Ok(true)
  }

  /// Fetches all the subobject IDs of an object via its monomorphisms.
  pub fn fetch_subobjects(&self) -> Result<HashMap<usize, Vec<(usize, usize)>>, String> {
    let mut subobjects: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for morphs in &self.morphisms {
      for morph in 0..morphs.1.len() {
        if self.is_monic(morphs.0 .0, morphs.0 .1, morph)? {
          subobjects.entry(morphs.0 .1).or_default().push((morphs.0 .0, morph));
        }
      }
    }
    Ok(subobjects)
  }

  /// Fetched the object ID for the terminal object
  pub fn terminal(&self) -> Result<usize, String> {
    let mut options = Vec::new();
    'outer: for (idx, _) in self.objects.iter().enumerate() {
      for (j, _) in self.objects.iter().enumerate() {
        let homset = self.morphisms.get(&(j, idx));
        match homset {
          None => continue 'outer,
          Some(morphs) if morphs.len() != 1 => continue 'outer,
          _ => {},
        }
      }
      options.push(idx);
    }
    match options.len() {
      0 => Err("No terminal object found in this category".to_string()),
      1 => Ok(options[0]),
      _ => Err(format!("Multiple terminal objects found: {options:?}")),
    }
  }
}

/// Builder struct for generating a category with a full first order set of power objects.
pub struct CategoryBuilder<O: Object, P: PowerObjectGenerator<O>> {
  objects:           Vec<O>,
  morphisms:         HashMap<(usize, usize), HomSet<O, O>>,
  pobject_generator: Option<P>,
}

impl<O: Object, P: PowerObjectGenerator<O>> CategoryBuilder<O, P> {
  /// Spawns a new builder
  pub fn new(objects: Vec<O>, morphisms: HashMap<(usize, usize), HomSet<O, O>>) -> Self {
    Self { objects, morphisms, pobject_generator: None }
  }

  /// Set the generator for power objects.
  /// If you aim to generate the full set of power objects upon build this must be set.
  pub fn set_generator(&mut self, pobj_generator: P) {
    self.pobject_generator = Some(pobj_generator);
  }

  /// Construct the category
  pub fn build(self) -> Category<O, P> {
    let mut pobjects = HashMap::new();
    let mut objects = self.objects.clone();
    let mut morphs = self.morphisms;
    if self.pobject_generator.is_some() {
      let gen = self.pobject_generator.as_ref().unwrap();
      for (i, _) in self.objects.iter().enumerate() {
        for (j, _) in self.objects.iter().enumerate() {
          let variants = [
            PowerObjectType::Coproduct(i, j),
            PowerObjectType::Exponential(i, j),
            PowerObjectType::Product(i, j),
          ];

          for var in variants {
            let (obj, homset) = gen.generate_power_object(&var, &objects);
            for morph in homset {
              let dom = morph.domain();
              let cod = morph.codomain();
              let (sidx, _) = objects.iter().enumerate().find(|&(_, x)| x == dom).unwrap();
              let (eidx, _) = objects.iter().enumerate().find(|&(_, x)| x == cod).unwrap();
              if morphs.contains_key(&(sidx, eidx)) {
                morphs.get_mut(&(sidx, eidx)).unwrap().push(morph);
                continue;
              }
              morphs.insert((sidx, eidx), vec![morph]);
            }
            objects.push(obj);
            pobjects.insert(var, objects.len() - 1);
          }
        }
      }
    }

    Category::from_builder(objects, morphs, Some(pobjects), self.pobject_generator)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // A toy object, identified just by a short &str label
  #[derive(Clone, PartialEq, Eq, Hash, Debug)]
  struct SetObj(&'static str);
  impl Object for SetObj {}

  // Identity morphism  id_A : A → A
  struct IdMorphism {
    obj: SetObj,
  }
  impl Morphism for IdMorphism {
    type Codomain = SetObj;
    type Domain = SetObj;

    fn domain(&self) -> &Self::Domain { &self.obj }

    fn codomain(&self) -> &Self::Codomain { &self.obj }

    fn map(&self, d: &Self::Domain) -> Self::Codomain { d.clone() }
  }

  // Constant morphism  const_{B→C} : B → C  that always yields `codomain`
  struct ConstMorphism {
    domain:   SetObj,
    codomain: SetObj,
  }
  impl Morphism for ConstMorphism {
    type Codomain = SetObj;
    type Domain = SetObj;

    fn domain(&self) -> &Self::Domain { &self.domain }

    fn codomain(&self) -> &Self::Codomain { &self.codomain }

    fn map(&self, _d: &Self::Domain) -> Self::Codomain { self.codomain.clone() }
  }

  // Stub power-object generator (unused in these tests)
  struct NoGen;
  impl PowerObjectGenerator<SetObj> for NoGen {
    fn generate_power_object(
      &self,
      _t: &PowerObjectType,
      _objs: &[SetObj],
    ) -> (SetObj, Vec<Box<dyn Morphism<Domain = SetObj, Codomain = SetObj>>>) {
      unreachable!("power objects not needed for these unit tests")
    }
  }

  // `check_eq_morphisms` should see two identical identities as equal
  #[test]
  fn identity_equality() {
    let a = SetObj("A");
    let id1: Box<dyn Morphism<Domain = _, Codomain = _>> = Box::new(IdMorphism { obj: a.clone() });
    let id2: Box<dyn Morphism<Domain = _, Codomain = _>> = Box::new(IdMorphism { obj: a.clone() });

    assert!(check_eq_morphisms(id1.as_ref(), id2.as_ref()));
  }

  // Composition f ; g should yield the expected result
  #[test]
  fn compose_constant_chain() {
    let a = SetObj("A");
    let b = SetObj("B");
    let c = SetObj("C");

    let f: Box<dyn Morphism<Domain = _, Codomain = _>> =
      Box::new(ConstMorphism { domain: a.clone(), codomain: b.clone() });
    let g: Box<dyn Morphism<Domain = _, Codomain = _>> =
      Box::new(ConstMorphism { domain: b.clone(), codomain: c.clone() });

    let result = compose(&a, f.as_ref(), g.as_ref());
    assert_eq!(result, c);
  }

  // Adding the *same* morphism twice shouldn’t duplicate it in the hom-set
  #[test]
  fn add_morphism_deduplicates() {
    let mut cat: Category<SetObj, NoGen> = Category::create(None);

    let a = SetObj("A");
    let b = SetObj("B");
    cat.add_object(a.clone()); // idx 0
    cat.add_object(b.clone()); // idx 1

    let f: Box<dyn Morphism<Domain = _, Codomain = _>> =
      Box::new(ConstMorphism { domain: a.clone(), codomain: b.clone() });

    // first insertion
    cat.add_morphism(0, 1, f).unwrap();
    // second insertion of an *equal* morphism
    let dup: Box<dyn Morphism<Domain = _, Codomain = _>> =
      Box::new(ConstMorphism { domain: a.clone(), codomain: b.clone() });
    cat.add_morphism(0, 1, dup).unwrap();

    // hom-set <0,1> should still contain exactly one morphism
    let homset = cat.morphisms.get(&(0, 1)).expect("hom-set must exist");
    assert_eq!(homset.len(), 1);
  }

  // Identity morphism `id_B` should be recognised as monic
  #[test]
  fn identity_is_monic() {
    let mut cat: Category<SetObj, NoGen> = Category::create(None);

    // Objects
    let a = SetObj("A");
    let b = SetObj("B");
    cat.add_object(a.clone()); // idx 0
    cat.add_object(b.clone()); // idx 1

    // Morphisms A → B (two different constants, to satisfy the inner loop)
    let f1: Box<dyn Morphism<Domain = _, Codomain = _>> =
      Box::new(ConstMorphism { domain: a.clone(), codomain: b.clone() });
    let f2: Box<dyn Morphism<Domain = _, Codomain = _>> =
      Box::new(ConstMorphism { domain: a.clone(), codomain: b.clone() /* same codomain */ });

    cat.add_morphism(0, 1, f1).unwrap();
    cat.add_morphism(0, 1, f2).unwrap();

    // Identity on B
    let id_b: Box<dyn Morphism<Domain = _, Codomain = _>> = Box::new(IdMorphism { obj: b.clone() });
    cat.add_morphism(1, 1, id_b).unwrap();

    assert!(cat.is_monic(1, 1, 0 /* idx of id_B in hom-set (1,1) */).unwrap());
  }

  // A tiny numeric object so we can manufacture arbitrary “fresh” values.
  #[derive(Clone, PartialEq, Eq, Hash, Debug)]
  struct NObj(usize);
  impl Object for NObj {}

  // Dumb power-object generator:
  // * Returns a brand-new `NObj` whose value encodes the variant type + indices
  // * Produces **no** extra morphisms (that isn’t the focus of this test)
  struct DummyGen;
  impl PowerObjectGenerator<NObj> for DummyGen {
    fn generate_power_object(
      &self,
      t: &PowerObjectType,
      _objs: &[NObj],
    ) -> (NObj, Vec<Box<dyn Morphism<Domain = NObj, Codomain = NObj>>>) {
      let tag = match t {
        PowerObjectType::Product(i, j) => 1_00 + i * 10 + j,
        PowerObjectType::Coproduct(i, j) => 2_00 + i * 10 + j,
        PowerObjectType::Exponential(i, j) => 3_00 + i * 10 + j,
      };
      (NObj(tag), Vec::new())
    }
  }

  #[test]
  fn builder_with_power_objects() {
    // Two seed objects ⇒ 2×2 pairs × 3 variants = 12 power objects
    let seed = vec![NObj(0), NObj(1)];

    let mut builder = CategoryBuilder::new(seed, HashMap::new());
    builder.set_generator(DummyGen);
    let mut cat = builder.build();

    // 2 originals + 12 generated = 14 objects
    assert_eq!(cat.objects.len(), 14);

    // power-object table must hold exactly 12 entries
    {
      // ⚠ inner scope → immutable borrow ends here
      let pmap = cat.power_objects.as_ref().expect("table present");
      assert_eq!(pmap.len(), 12);
    } // <-- immutable borrow of `cat` ends

    // Now it's safe to mutably borrow `cat`
    let idx =
      cat.fetch_power_object_id(PowerObjectType::Product(0, 1)).expect("must exist already");

    // Re-borrow immutably just for the check
    let recorded =
      *cat.power_objects.as_ref().unwrap().get(&PowerObjectType::Product(0, 1)).unwrap();

    assert_eq!(idx, recorded);
  }
}
