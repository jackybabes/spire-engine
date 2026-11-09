//! Potions: static data (generated). Effects are applied in combat.

mod data;

pub use data::{PotionId, DEFECT_POTION_POOL, IRONCLAD_POTION_POOL, POTION_DATA, SILENT_POTION_POOL, WATCHER_POTION_POOL};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PotionRarity {
    Placeholder,
    Common,
    Uncommon,
    Rare,
}

#[derive(Clone, Copy, Debug)]
pub struct PotionData {
    pub id: PotionId,
    pub java_id: &'static str,
    pub name: &'static str,
    pub rarity: PotionRarity,
    pub potency: i32,
    pub needs_target: bool,
}

impl PotionId {
    pub fn data(self) -> &'static PotionData {
        &POTION_DATA[self as usize]
    }

    pub fn java_id(self) -> &'static str {
        self.data().java_id
    }

    pub fn from_java_id(id: &str) -> Option<PotionId> {
        POTION_DATA.iter().find(|d| d.java_id == id).map(|d| d.id)
    }

    /// Looks a potion up by game id, display name or variant name (case and spacing ignored).
    pub fn from_name(name: &str) -> Option<PotionId> {
        let norm = |s: &str| s.to_ascii_lowercase().replace([' ', '_', '\'', '-'], "");
        let n = norm(name);
        POTION_DATA.iter().find(|d| norm(d.java_id) == n || norm(d.name) == n || norm(&format!("{:?}", d.id)) == n).map(|d| d.id)
    }
}
