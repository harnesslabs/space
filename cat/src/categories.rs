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
  first: &Box<dyn Morphism<Domain = A, Codomain = B>>,
  second: &Box<dyn Morphism<Domain = A, Codomain = B>>,
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
  first: &Box<dyn Morphism<Domain = A, Codomain = B>>,
  second: &Box<dyn Morphism<Domain = B, Codomain = C>>,
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
  fn from_builder(
    objects: Vec<O>,
    morphisms: HashMap<(usize, usize), HomSet<O, O>>,
    power_objects: Option<HashMap<PowerObjectType, usize>>,
    generator: Option<P>,
  ) -> Self {
    Self { objects, morphisms, power_objects, generator }
  }

  pub fn create(generator: Option<P>) -> Self {
    let power_objects = if generator.is_none() { None } else { Some(HashMap::new()) };
    Category { objects: Vec::new(), morphisms: HashMap::new(), power_objects, generator }
  }

  pub fn from_object_list(objects: &[O], generator: Option<P>) -> Self {
    let power_objects = if generator.is_none() { None } else { Some(HashMap::new()) };
    Category { objects: objects.to_vec(), morphisms: HashMap::new(), power_objects, generator }
  }

  pub fn add_object(&mut self, object: O) {
    if !self.objects.contains(&object) {
      self.objects.push(object)
    }
  }

  pub fn add_morphism(
    &mut self,
    domain: usize,
    codomain: usize,
    map: Box<dyn Morphism<Domain = O, Codomain = O>>,
  ) -> Result<(), String> {
    let actual_domain = map.domain();
    let actual_codomain = map.codomain();

    if self.objects.get(domain) != Some(actual_domain) {
      return Err(format!("Domain index {} does not match morphism's domain object", domain));
    }
    if self.objects.get(codomain) != Some(actual_codomain) {
      return Err(format!("Codomain index {} does not match morphism's codomain object", codomain));
    }

    let key = (domain, codomain);
    if let Some(homset) = self.morphisms.get_mut(&key) {
      let mut insert = true;
      for m in homset.iter() {
        if check_eq_morphisms(m, &map) {
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

pub struct CategoryBuilder<O: Object, P: PowerObjectGenerator<O>> {
  objects:           Vec<O>,
  morphisms:         HashMap<(usize, usize), HomSet<O, O>>,
  pobject_generator: Option<P>,
}

impl<O: Object, P: PowerObjectGenerator<O>> CategoryBuilder<O, P> {
  pub fn new(objects: Vec<O>, morphisms: HashMap<(usize, usize), HomSet<O, O>>) -> Self {
    Self { objects, morphisms, pobject_generator: None }
  }

  pub fn set_generator(&mut self, pobj_generator: P) {
    self.pobject_generator = Some(pobj_generator);
  }

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
