//! Card, relic and potion pools and the reward rolls that draw from them
//! (`AbstractDungeon.initializeCardPools`, `initializeRelicList`, `getRewardCards`,
//! `returnRandomRelic`, `returnRandomPotion`, ...).

use serde::{Deserialize, Serialize};
use sts_rng::{java_shuffle, JavaRandom, StsRandom};

use crate::cards::{CardId, CardRarity, CardType, COLORLESS_LIBRARY_ORDER};
use crate::character::Character;
use crate::potions::{PotionId, PotionRarity};
use crate::relics::{RelicId, RelicState, RelicTier, SHARED_LIBRARY_ORDER};

/// `CardLibrary.curses` iteration order (a separate HashMap from `cards`).
pub const CURSE_MAP_ORDER: [CardId; 14] = [
    CardId::Regret,
    CardId::Necronomicurse,
    CardId::Injury,
    CardId::Shame,
    CardId::Parasite,
    CardId::Normality,
    CardId::CurseOfTheBell,
    CardId::Doubt,
    CardId::Writhe,
    CardId::Pain,
    CardId::Pride,
    CardId::Decay,
    CardId::AscendersBane,
    CardId::Clumsy,
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pools {
    pub common: Vec<CardId>,
    pub uncommon: Vec<CardId>,
    pub rare: Vec<CardId>,
    pub colorless: Vec<CardId>,
    pub common_relics: Vec<RelicId>,
    pub uncommon_relics: Vec<RelicId>,
    pub rare_relics: Vec<RelicId>,
    pub shop_relics: Vec<RelicId>,
    pub boss_relics: Vec<RelicId>,
}

/// Where a card reward is rolled; sets the base rarity chances (`AbstractRoom.getCardRarity`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardRoom {
    Normal,
    Elite,
    Boss,
    Shop,
    Rest,
}

impl Pools {
    /// `initializeCardPools` (the character's cards, everything unlocked).
    pub fn new_cards(character: Character) -> (Vec<CardId>, Vec<CardId>, Vec<CardId>, Vec<CardId>) {
        let by = |r: CardRarity| character.library_order().iter().copied().filter(|c| c.data().rarity == r).collect::<Vec<_>>();
        (by(CardRarity::Common), by(CardRarity::Uncommon), by(CardRarity::Rare), COLORLESS_LIBRARY_ORDER.to_vec())
    }

    /// `initializeRelicList`: build each tier from the library maps (shared, then the
    /// character's), then shuffle with relicRng.
    pub fn new(relic_rng: &mut StsRandom, owned: &[RelicState], character: Character) -> Pools {
        let tier = |t: RelicTier| {
            SHARED_LIBRARY_ORDER.iter().chain(character.relic_library_order()).copied().filter(|r| r.data().tier == t).collect::<Vec<_>>()
        };
        let mut lists = [tier(RelicTier::Common), tier(RelicTier::Uncommon), tier(RelicTier::Rare), tier(RelicTier::Shop), tier(RelicTier::Boss)];
        for l in lists.iter_mut() {
            let seed = relic_rng.random_long();
            java_shuffle(l, &mut JavaRandom::new(seed));
        }
        for r in owned {
            for l in lists.iter_mut() {
                if let Some(i) = l.iter().position(|x| *x == r.id) {
                    l.remove(i);
                }
            }
        }
        let [common_relics, uncommon_relics, rare_relics, shop_relics, boss_relics] = lists;
        let (common, uncommon, rare, colorless) = Pools::new_cards(character);
        Pools { common, uncommon, rare, colorless, common_relics, uncommon_relics, rare_relics, shop_relics, boss_relics }
    }

    pub fn card_pool(&self, r: CardRarity) -> &Vec<CardId> {
        match r {
            CardRarity::Common => &self.common,
            CardRarity::Uncommon => &self.uncommon,
            _ => &self.rare,
        }
    }

    /// `srcXCardPool`: the rarity pool in reverse.
    pub fn src_pool(&self, r: CardRarity) -> Vec<CardId> {
        self.card_pool(r).iter().rev().copied().collect()
    }

    /// `getCard(rarity)` / `getCard(rarity, rng)`: `pool.getRandomCard(rng)`.
    pub fn random_card(&self, r: CardRarity, rng: &mut StsRandom) -> CardId {
        let p = self.card_pool(r);
        p[rng.random(p.len() as i32 - 1) as usize]
    }

    /// `CardLibrary.getAnyColorCard(rarity)` (Prismatic Shard's rewards): every library card of
    /// that rarity (no curses or statuses), shuffled with cardRng, then sorted by card ID for
    /// `getRandomCard(true, rarity)`, which picks with cardRng too.
    pub fn any_color_card(r: CardRarity, card_rng: &mut StsRandom) -> CardId {
        let mut tmp: Vec<CardId> =
            crate::cards::ALL_CARDS.iter().copied().filter(|c| c.data().rarity == r && !matches!(c.data().card_type, CardType::Curse | CardType::Status)).collect();
        let _ = card_rng.random_long();
        tmp.sort_by(|a, b| a.data().java_id.cmp(b.data().java_id));
        tmp[card_rng.random(tmp.len() as i32 - 1) as usize]
    }

    /// `colorlessCardPool.getRandomCard(true, rarity)`: filtered, sorted by card ID, cardRng.
    pub fn random_colorless(&self, r: CardRarity, card_rng: &mut StsRandom) -> Option<CardId> {
        let mut tmp: Vec<CardId> = self.colorless.iter().copied().filter(|c| c.data().rarity == r).collect();
        if tmp.is_empty() {
            return None;
        }
        tmp.sort_by(|a, b| a.data().java_id.cmp(b.data().java_id));
        Some(tmp[card_rng.random(tmp.len() as i32 - 1) as usize])
    }

    fn pool_mut(&mut self, t: RelicTier) -> &mut Vec<RelicId> {
        match t {
            RelicTier::Common => &mut self.common_relics,
            RelicTier::Uncommon => &mut self.uncommon_relics,
            RelicTier::Rare => &mut self.rare_relics,
            RelicTier::Shop => &mut self.shop_relics,
            _ => &mut self.boss_relics,
        }
    }

    /// `returnRandomRelicKey(tier)`: pop from the front, falling back to the next tier when empty.
    pub fn take_relic(&mut self, t: RelicTier, ctx: &SpawnCtx) -> RelicId {
        let pick = match t {
            RelicTier::Common if self.common_relics.is_empty() => return self.take_relic(RelicTier::Uncommon, ctx),
            RelicTier::Uncommon if self.uncommon_relics.is_empty() => return self.take_relic(RelicTier::Rare, ctx),
            RelicTier::Rare if self.rare_relics.is_empty() => RelicId::Circlet,
            RelicTier::Shop if self.shop_relics.is_empty() => return self.take_relic(RelicTier::Uncommon, ctx),
            RelicTier::Boss if self.boss_relics.is_empty() => RelicId::RedCirclet,
            _ => self.pool_mut(t).remove(0),
        };
        if can_spawn(pick, ctx) {
            pick
        } else {
            self.take_relic_end(t, ctx)
        }
    }

    /// `returnEndRandomRelicKey(tier)`: pop from the back (the front for boss relics).
    pub fn take_relic_end(&mut self, t: RelicTier, ctx: &SpawnCtx) -> RelicId {
        let pick = match t {
            RelicTier::Common if self.common_relics.is_empty() => return self.take_relic(RelicTier::Uncommon, ctx),
            RelicTier::Uncommon if self.uncommon_relics.is_empty() => return self.take_relic(RelicTier::Rare, ctx),
            RelicTier::Rare if self.rare_relics.is_empty() => RelicId::Circlet,
            RelicTier::Shop if self.shop_relics.is_empty() => return self.take_relic(RelicTier::Uncommon, ctx),
            RelicTier::Boss if self.boss_relics.is_empty() => RelicId::RedCirclet,
            RelicTier::Boss => self.boss_relics.remove(0),
            _ => self.pool_mut(t).pop().unwrap(),
        };
        if can_spawn(pick, ctx) {
            pick
        } else {
            self.take_relic_end(t, ctx)
        }
    }
}

/// What `canSpawn()` overrides look at.
pub struct SpawnCtx<'a> {
    pub floor: i32,
    pub act: i32,
    pub in_shop: bool,
    pub relics: &'a [RelicState],
    pub deck: &'a [crate::cards::Card],
}

/// `AbstractRelic.canSpawn()` and its overrides (non-endless).
pub fn can_spawn(r: RelicId, ctx: &SpawnCtx) -> bool {
    use RelicId::*;
    let has = |id: RelicId| ctx.relics.iter().any(|x| x.id == id);
    let f = ctx.floor;
    match r {
        AncientTeaSet | CeramicFish | DreamCatcher | DarkstonePeriapt | FrozenEgg2 | JuzuBracelet | MeatOnTheBone | MealTicket
        | MoltenEgg2 | Omamori | PrayerWheel | PotionBelt | QuestionCard | RegalPillow | SingingBowl | ToxicEgg2 => f <= 48,
        Courier | MawBank | OldCoin | SmilingMask => f <= 48 && !ctx.in_shop,
        Matryoshka | WingBoots => f <= 40,
        PreservedInsect => f <= 52,
        TinyChest => f <= 35,
        BlackBlood => has(BurningBlood),
        FrozenCore => has(CrackedCore),
        HolyWater => has(PureWater),
        RingOfTheSerpent => has(SnakeRing),
        Ectoplasm => ctx.act <= 1,
        BottledFlame => ctx.deck.iter().any(|c| c.card_type() == CardType::Attack && c.data().rarity != CardRarity::Basic),
        BottledLightning => ctx.deck.iter().any(|c| c.card_type() == CardType::Skill && c.data().rarity != CardRarity::Basic),
        BottledTornado => ctx.deck.iter().any(|c| c.card_type() == CardType::Power),
        Girya | PeacePipe | Shovel => f < 48 && ctx.relics.iter().filter(|x| matches!(x.id, PeacePipe | Shovel | Girya)).count() < 2,
        _ => true,
    }
}

/// `PotionHelper.getRandomPotion()` from the character's pool.
pub fn random_potion(character: Character, potion_rng: &mut StsRandom) -> PotionId {
    let pool = character.potion_pool();
    pool[potion_rng.random(pool.len() as i32 - 1) as usize]
}

/// `AbstractDungeon.returnRandomPotion()`: roll a rarity, then draw until it matches.
pub fn return_random_potion(character: Character, potion_rng: &mut StsRandom) -> PotionId {
    return_random_potion_limited(character, potion_rng, false)
}

/// `returnRandomPotion(limited)`: `limited` also rejects Fruit Juice.
pub fn return_random_potion_limited(character: Character, potion_rng: &mut StsRandom, limited: bool) -> PotionId {
    let roll = potion_rng.random_range(0, 99);
    let rarity = if roll < 65 {
        PotionRarity::Common
    } else if roll < 90 {
        PotionRarity::Uncommon
    } else {
        PotionRarity::Rare
    };
    // `while (temp.rarity != rarity || spamCheck)`, with spamCheck cleared by any draw that is
    // not Fruit Juice.
    let mut p = random_potion(character, potion_rng);
    let mut spam = limited;
    while p.data().rarity != rarity || spam {
        spam = limited;
        p = random_potion(character, potion_rng);
        if p != PotionId::FruitJuice {
            spam = false;
        }
    }
    p
}
