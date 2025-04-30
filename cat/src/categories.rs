use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

pub trait Object: Clone + Eq + Hash + Debug {}

pub trait Morphism {
    type Domain: Object + ?Sized;
    type Codomain: Object + ?Sized;
    fn domain(&self) -> &Self::Domain;
    fn codomain(&self) -> &Self::Codomain;
    fn map(&self, domain: &Self::Domain) -> Self::Codomain;
}

pub fn check_eq_morphisms<A: Object, B: Object>(first: &Box<dyn Morphism<Domain = A, Codomain = B>>, second: &Box<dyn Morphism<Domain = A, Codomain = B>>) -> bool {
    if first.domain() == second.domain() && first.codomain() == second.codomain() && first.map(first.domain()) == second.map(second.domain()) {
        return true
    }
    false
}

pub fn compose<A: Object, B: Object, C: Object>(domain: &A, first: &Box<dyn Morphism<Domain = A, Codomain = B>>, second: &Box<dyn Morphism<Domain = B, Codomain = C>>) -> C {
    second.map(&first.map(domain))
}

type HomSet<A, B> = Vec<Box<dyn Morphism<Domain = A, Codomain = B>>>;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum PowerObjectType {
    Product(usize, usize),
    Coproduct(usize, usize),
    Exponential(usize, usize),
}

pub trait PowerObjectGenerator<O: Object + ?Sized> {
    fn generate_power_object(
        &self,
        power_type: &PowerObjectType,
        objects: &[O],
    ) -> Option<(O, Vec<Box<dyn Morphism<Domain = O, Codomain = O>>>)>;
}

/// A `Category` of a single class of object
/// e.g. Vect_k, Hilb_k
pub struct Category<O: Object + ?Sized> {
    objects: Vec<O>,
    morphisms: HashMap<(usize, usize), HomSet<O, O>>,
    power_objects: HashMap<PowerObjectType, usize>,
    generator: Box<dyn PowerObjectGenerator<O>>,
}

impl<O: Object + ?Sized> Category<O> {
    pub fn create(generator: Box<dyn PowerObjectGenerator<O>>) -> Self {
        Category { objects: Vec::new(), morphisms: HashMap::new(), power_objects: HashMap::new(), generator }
    }
    pub fn from_object_list(objects: &[O], generator: Box<dyn PowerObjectGenerator<O>>) -> Self {
        Category { objects: objects.to_vec(), morphisms: HashMap::new(), power_objects: HashMap::new(), generator }
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

    pub fn get_power_object(&mut self, power_type: PowerObjectType) -> Result<usize, String> {
        if let Some(&idx) = self.power_objects.get(&power_type) {
            return Ok(idx);
        }

        let (new_obj, new_morphisms) = self.generator
            .generate_power_object(&power_type, &self.objects)
            .ok_or_else(|| format!("Failed to generate power object {:?}", power_type))?;

        self.add_object(new_obj);
        let new_idx = self.objects.len() - 1;
        self.power_objects.insert(power_type, new_idx);

        for morph in new_morphisms {
            let domain_obj = morph.domain();
            let codomain_obj = morph.codomain();

            let domain_idx = self.objects.iter()
                .position(|o| o == domain_obj)
                .ok_or_else(|| format!("Domain object not found for morphism"))?;
            let codomain_idx = self.objects.iter()
                .position(|o| o == codomain_obj)
                .ok_or_else(|| format!("Codomain object not found for morphism"))?;

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
            let vecs = homset.unwrap();
            let results = vecs.iter().map(|x| base.map(&x.map(x.domain()))).collect::<Vec<_>>();
            let mut seen = HashSet::new();
            for item in results {
                if !seen.insert(item) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    pub fn fetch_subobjects(&self) -> Result<HashMap<usize, Vec<(usize, usize)>>, String> {
        let mut subobjects: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for morphs in &self.morphisms {
            for morph in 0..morphs.1.len() {
                if self.is_monic(morphs.0.0, morphs.0.1, morph)? {
                    subobjects
                            .entry(morphs.0.1)
                            .or_insert_with(Vec::new)
                            .push((morphs.0.0, morph));
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
                    None => {
                        continue 'outer
                    }
                    Some(morphs) if morphs.len() != 1 => {
                        continue 'outer
                    }
                    _ => {}
                }
            }
            options.push(idx);
        }
        match options.len() {
            0 => Err("No terminal object found in this category".to_string()),
            1 => Ok(options[0]),
            _ => Err(format!("Multiple terminal objects found: {:?}", options))
        }
    }
}