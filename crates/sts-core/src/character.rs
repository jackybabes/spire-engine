//! The playable characters (`AbstractPlayer.PlayerClass`) and what differs between them.

use serde::{Deserialize, Serialize};

use crate::cards::{CardColor, CardId, BLUE_LIBRARY_ORDER, GREEN_LIBRARY_ORDER, PURPLE_LIBRARY_ORDER, RED_LIBRARY_ORDER};
use crate::potions::{PotionId, DEFECT_POTION_POOL, IRONCLAD_POTION_POOL, SILENT_POTION_POOL, WATCHER_POTION_POOL};
use crate::relics::{self, RelicId};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Character {
    #[default]
    Ironclad,
    Silent,
    Defect,
    Watcher,
}

impl Character {
    pub fn card_color(self) -> CardColor {
        match self {
            Character::Ironclad => CardColor::Red,
            Character::Silent => CardColor::Green,
            Character::Defect => CardColor::Blue,
            Character::Watcher => CardColor::Purple,
        }
    }

    /// The character's cards in `CardLibrary` order (what `addRedCards` and friends yield).
    pub fn library_order(self) -> &'static [CardId] {
        match self {
            Character::Ironclad => &RED_LIBRARY_ORDER,
            Character::Silent => &GREEN_LIBRARY_ORDER,
            Character::Defect => &BLUE_LIBRARY_ORDER,
            Character::Watcher => &PURPLE_LIBRARY_ORDER,
        }
    }

    /// `getStartingDeck`.
    pub fn starting_deck(self) -> Vec<CardId> {
        use CardId::*;
        let (strike, defend, rest): (CardId, CardId, &[CardId]) = match self {
            Character::Ironclad => (StrikeR, DefendR, &[Bash]),
            Character::Silent => (StrikeG, DefendG, &[Survivor, Neutralize]),
            Character::Defect => (StrikeB, DefendB, &[Zap, Dualcast]),
            Character::Watcher => (StrikeP, DefendP, &[Eruption, Vigilance]),
        };
        let (n_strike, n_defend) = match self {
            Character::Ironclad => (5, 4),
            Character::Silent => (5, 5),
            Character::Defect | Character::Watcher => (4, 4),
        };
        let mut deck = vec![strike; n_strike];
        deck.extend(vec![defend; n_defend]);
        deck.extend_from_slice(rest);
        deck
    }

    /// `getStartingRelics`.
    pub fn starting_relic(self) -> RelicId {
        match self {
            Character::Ironclad => RelicId::BurningBlood,
            Character::Silent => RelicId::SnakeRing,
            Character::Defect => RelicId::CrackedCore,
            Character::Watcher => RelicId::PureWater,
        }
    }

    /// The character's relics in `RelicLibrary` order (`redRelics`, `greenRelics`... iteration).
    pub fn relic_library_order(self) -> &'static [RelicId] {
        match self {
            Character::Ironclad => &relics::RED_LIBRARY_ORDER,
            Character::Silent => &relics::GREEN_LIBRARY_ORDER,
            Character::Defect => &relics::BLUE_LIBRARY_ORDER,
            Character::Watcher => &relics::PURPLE_LIBRARY_ORDER,
        }
    }

    /// `PotionHelper.getPotions(class, false)`: the character's three potions, then the shared ones.
    pub fn potion_pool(self) -> &'static [PotionId] {
        match self {
            Character::Ironclad => &IRONCLAD_POTION_POOL,
            Character::Silent => &SILENT_POTION_POOL,
            Character::Defect => &DEFECT_POTION_POOL,
            Character::Watcher => &WATCHER_POTION_POOL,
        }
    }

    /// `getStartCardForEvent` (Match and Keep).
    pub fn start_card_for_event(self) -> CardId {
        match self {
            Character::Ironclad => CardId::Bash,
            Character::Silent => CardId::Neutralize,
            Character::Defect => CardId::Zap,
            Character::Watcher => CardId::Eruption,
        }
    }

    /// `getAscensionMaxHPLoss` (Ascension 14).
    pub fn ascension_max_hp_loss(self) -> i32 {
        match self {
            Character::Ironclad => 5,
            _ => 4,
        }
    }

    /// The name `sts-cli` and the Python bindings take (`ironclad`, `silent`, `defect`, `watcher`).
    pub fn from_name(name: &str) -> Option<Character> {
        match name.to_ascii_lowercase().as_str() {
            "ironclad" | "red" => Some(Character::Ironclad),
            "silent" | "the_silent" | "green" => Some(Character::Silent),
            "defect" | "blue" => Some(Character::Defect),
            "watcher" | "purple" => Some(Character::Watcher),
            _ => None,
        }
    }

    /// The lowercase name `from_name` reads and CommunicationMod's `start` command takes.
    pub fn name(self) -> &'static str {
        match self {
            Character::Ironclad => "ironclad",
            Character::Silent => "silent",
            Character::Defect => "defect",
            Character::Watcher => "watcher",
        }
    }

    /// `AbstractPlayer.PlayerClass` names, as the game's state and run files write them.
    pub fn class_name(self) -> &'static str {
        match self {
            Character::Ironclad => "IRONCLAD",
            Character::Silent => "THE_SILENT",
            Character::Defect => "DEFECT",
            Character::Watcher => "WATCHER",
        }
    }

    /// `masterMaxOrbs`: the Defect starts every fight with 3 orb slots.
    pub fn orb_slots(self) -> usize {
        match self {
            Character::Defect => 3,
            _ => 0,
        }
    }

    /// `getLoadout().maxHp` (before Ascension 14's cut).
    pub fn max_hp(self) -> i32 {
        match self {
            Character::Ironclad => 80,
            Character::Silent => 70,
            Character::Defect => 75,
            Character::Watcher => 72,
        }
    }
}
