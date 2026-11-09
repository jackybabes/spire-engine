//! Neow's blessing (`NeowEvent`, `NeowReward`).

use serde::{Deserialize, Serialize};
use sts_rng::StsRandom;

use super::pools::{self, RewardRoom, CURSE_MAP_ORDER};
use super::relics::TransformRng;
use super::{GridPurpose, Reward, Run, Screen};
use crate::cards::{CardColor, CardId, CardRarity};
use crate::relics::{RelicId, RelicTier};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeowReward {
    RandomColorless2,
    ThreeCards,
    OneRandomRareCard,
    RemoveCard,
    UpgradeCard,
    RandomColorless,
    TransformCard,
    ThreeSmallPotions,
    RandomCommonRelic,
    TenPercentHpBonus,
    HundredGold,
    ThreeEnemyKill,
    RemoveTwo,
    TransformTwoCards,
    OneRareRelic,
    ThreeRareCards,
    TwoFiftyGold,
    TwentyPercentHpBonus,
    BossRelic,
}

impl NeowReward {
    /// Every Neow reward, in declaration order (the order the encodings number them).
    pub fn all() -> impl Iterator<Item = NeowReward> {
        // SAFETY: a field-less enum with contiguous discriminants 0..=BossRelic; transmute checks
        // at compile time that it is one byte.
        (0..=NeowReward::BossRelic as u8).map(|i| unsafe { std::mem::transmute::<u8, NeowReward>(i) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeowDrawback {
    None,
    TenPercentHpLoss,
    NoGold,
    Curse,
    PercentDamage,
}

impl NeowDrawback {
    /// Every Neow drawback, in declaration order (the order the encodings number them).
    pub fn all() -> impl Iterator<Item = NeowDrawback> {
        // SAFETY: a field-less enum with contiguous discriminants 0..=PercentDamage; transmute checks
        // at compile time that it is one byte.
        (0..=NeowDrawback::PercentDamage as u8).map(|i| unsafe { std::mem::transmute::<u8, NeowDrawback>(i) })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeowOption {
    pub reward: NeowReward,
    pub drawback: NeowDrawback,
}

/// `NeowEvent.blessing()` / `miniBlessing()`: roll the options.
pub(super) fn open(run: &mut Run) -> Screen {
    use NeowReward::*;
    let mut options = Vec::new();
    if !run.config.neow_full {
        options.push(NeowOption { reward: ThreeEnemyKill, drawback: NeowDrawback::None });
        options.push(NeowOption { reward: TenPercentHpBonus, drawback: NeowDrawback::None });
        return Screen::Neow { options };
    }
    let mut rng = StsRandom::new(run.config.seed);
    let pick = |rng: &mut StsRandom, list: &[NeowReward]| list[rng.random_range(0, list.len() as i32 - 1) as usize];
    options.push(NeowOption {
        reward: pick(&mut rng, &[ThreeCards, OneRandomRareCard, RemoveCard, UpgradeCard, TransformCard, RandomColorless]),
        drawback: NeowDrawback::None,
    });
    options.push(NeowOption {
        reward: pick(&mut rng, &[ThreeSmallPotions, RandomCommonRelic, TenPercentHpBonus, ThreeEnemyKill, HundredGold]),
        drawback: NeowDrawback::None,
    });
    let drawbacks = [NeowDrawback::TenPercentHpLoss, NeowDrawback::NoGold, NeowDrawback::Curse, NeowDrawback::PercentDamage];
    let drawback = drawbacks[rng.random_range(0, 3) as usize];
    let mut list = vec![RandomColorless2];
    if drawback != NeowDrawback::Curse {
        list.push(RemoveTwo);
    }
    list.push(OneRareRelic);
    list.push(ThreeRareCards);
    if drawback != NeowDrawback::NoGold {
        list.push(TwoFiftyGold);
    }
    list.push(TransformTwoCards);
    if drawback != NeowDrawback::TenPercentHpLoss {
        list.push(TwentyPercentHpBonus);
    }
    options.push(NeowOption { reward: pick(&mut rng, &list), drawback });
    // Category 3 has a single option but still rolls random(0, 0).
    options.push(NeowOption { reward: pick(&mut rng, &[BossRelic]), drawback: NeowDrawback::None });
    run.neow_rng = Some(rng);
    Screen::Neow { options }
}

fn hp_bonus(run: &Run) -> i32 {
    (run.max_hp as f32 * 0.1) as i32
}

/// `NeowReward.activate()` followed by the curse from `update()`.
pub(super) fn choose(run: &mut Run, opt: NeowOption) {
    use NeowReward::*;
    let bonus = hp_bonus(run);
    match opt.drawback {
        NeowDrawback::NoGold => run.gold = 0,
        NeowDrawback::TenPercentHpLoss => run.decrease_max_hp(bonus),
        NeowDrawback::PercentDamage => run.hp -= run.hp / 10 * 3,
        _ => {}
    }
    run.screen = Screen::Map;
    match opt.reward {
        UpgradeCard => grid(run, GridPurpose::Upgrade, (0..run.deck.len()).filter(|&i| run.deck[i].can_upgrade()).collect(), 1),
        RemoveCard => grid(run, GridPurpose::Remove, purgeable(run), 1),
        RemoveTwo => grid(run, GridPurpose::Remove, purgeable(run), 2),
        TransformCard => grid(run, GridPurpose::Transform, purgeable(run), 1),
        TransformTwoCards => grid(run, GridPurpose::Transform, purgeable(run), 2),
        RandomColorless2 => colorless_cards(run, true),
        RandomColorless => colorless_cards(run, false),
        ThreeRareCards => reward_cards(run, true),
        ThreeCards => reward_cards(run, false),
        HundredGold => run.gain_gold(100),
        TwoFiftyGold => run.gain_gold(250),
        OneRandomRareCard => {
            let mut rng = run.neow_rng.take().unwrap_or_else(|| StsRandom::new(run.config.seed));
            let id = run.pools.random_card(CardRarity::Rare, &mut rng);
            run.neow_rng = Some(rng);
            let c = run.new_card(id);
            run.obtain_card(c);
        }
        RandomCommonRelic => {
            let r = run.random_relic(RelicTier::Common);
            run.obtain_relic(r);
        }
        OneRareRelic => {
            let r = run.random_relic(RelicTier::Rare);
            run.obtain_relic(r);
        }
        BossRelic => {
            run.relics.remove(0);
            let r = run.random_relic(RelicTier::Boss);
            run.obtain_relic(r);
        }
        ThreeEnemyKill => run.obtain_relic(RelicId::NeowsLament),
        TenPercentHpBonus => run.increase_max_hp(bonus),
        TwentyPercentHpBonus => run.increase_max_hp(bonus * 2),
        ThreeSmallPotions => {
            let items: Vec<Reward> = (0..3).map(|_| Reward::Potion(pools::random_potion(run.config.character, &mut run.rng.potion))).collect();
            // CombatRewardScreen.open() rolls a card reward, which Neow then removes.
            let _ = run.reward_cards(RewardRoom::Normal);
            run.screen = Screen::Rewards { items };
        }
    }
    if opt.drawback == NeowDrawback::Curse {
        let id = random_curse(&mut run.rng.card);
        let c = run.new_card(id);
        run.obtain_card(c);
    }
}

fn grid(run: &mut Run, purpose: GridPurpose, options: Vec<usize>, count: usize) {
    if options.len() >= count {
        run.screen = Screen::Grid { purpose, options, count };
    }
}

/// `CardGroup.getPurgeableCards()`.
pub(super) fn purgeable(run: &Run) -> Vec<usize> {
    (0..run.deck.len())
        .filter(|&i| !matches!(run.deck[i].id, CardId::Necronomicurse | CardId::CurseOfTheBell | CardId::AscendersBane))
        .collect()
}

fn roll_rarity(rng: &mut StsRandom) -> CardRarity {
    if rng.random_boolean_chance(0.33) {
        CardRarity::Uncommon
    } else {
        CardRarity::Common
    }
}

/// `NeowReward.getRewardCards(rareOnly)`.
fn reward_cards(run: &mut Run, rare_only: bool) {
    let mut rng = run.neow_rng.take().unwrap_or_else(|| StsRandom::new(run.config.seed));
    let mut ids: Vec<CardId> = Vec::new();
    for _ in 0..3 {
        let mut rarity = roll_rarity(&mut rng);
        if rare_only {
            rarity = CardRarity::Rare;
        }
        let mut c = run.pools.random_card(rarity, &mut rng);
        while ids.contains(&c) {
            c = run.pools.random_card(rarity, &mut rng);
        }
        ids.push(c);
    }
    run.neow_rng = Some(rng);
    let cards = ids.into_iter().map(|id| run.new_card(id)).collect();
    run.screen = Screen::CardReward { item: None, cards };
}

/// `NeowReward.getColorlessRewardCards(rareOnly)`: rarity from Neow's rng, cards from cardRng.
fn colorless_cards(run: &mut Run, rare_only: bool) {
    let mut rng = run.neow_rng.take().unwrap_or_else(|| StsRandom::new(run.config.seed));
    let mut ids: Vec<CardId> = Vec::new();
    for _ in 0..3 {
        let mut rarity = roll_rarity(&mut rng);
        if rare_only {
            rarity = CardRarity::Rare;
        } else if rarity == CardRarity::Common {
            rarity = CardRarity::Uncommon;
        }
        let draw = |run: &mut Run| {
            run.pools
                .random_colorless(rarity, &mut run.rng.card)
                .or_else(|| run.pools.random_colorless(CardRarity::Uncommon, &mut run.rng.card))
                .expect("colorless pool is empty")
        };
        let mut c = draw(run);
        while ids.contains(&c) {
            c = draw(run);
        }
        ids.push(c);
    }
    run.neow_rng = Some(rng);
    let cards = ids.into_iter().map(|id| run.new_card(id)).collect();
    run.screen = Screen::CardReward { item: None, cards };
}

/// `CardLibrary.getCurse()`: a random curse, using cardRng.
pub fn random_curse(card_rng: &mut StsRandom) -> CardId {
    let list: Vec<CardId> = CURSE_MAP_ORDER
        .iter()
        .copied()
        .filter(|c| !matches!(c, CardId::AscendersBane | CardId::Necronomicurse | CardId::CurseOfTheBell | CardId::Pride))
        .collect();
    list[card_rng.random_range(0, list.len() as i32 - 1) as usize]
}

/// `AbstractDungeon.transformCard(c, false, rng)` for the cards at `indices`, which leave the deck.
pub(super) fn transform(run: &mut Run, indices: &[usize]) {
    let removed = run.take_cards(indices);
    for c in removed {
        let id = transformed_id(run, c.id, TransformRng::Neow);
        let card = run.new_card(id);
        run.obtain_card(card);
    }
}

/// The card a transform turns `from` into (`returnTrulyRandomCardFromAvailable` and friends).
pub(super) fn transformed_id(run: &mut Run, from: CardId, which: TransformRng) -> CardId {
    let list: Vec<CardId> = match from.data().color {
        CardColor::Curse => CURSE_MAP_ORDER
            .iter()
            .copied()
            .filter(|c| *c != from && !matches!(c, CardId::Necronomicurse | CardId::AscendersBane | CardId::CurseOfTheBell | CardId::Pride))
            .collect(),
        CardColor::Colorless => run.pools.colorless.iter().rev().copied().filter(|c| *c != from).collect(),
        _ => {
            let mut list: Vec<CardId> = run.pools.common.iter().copied().filter(|c| *c != from).collect();
            list.extend(run.pools.src_pool(CardRarity::Uncommon).into_iter().filter(|c| *c != from));
            list.extend(run.pools.src_pool(CardRarity::Rare).into_iter().filter(|c| *c != from));
            list
        }
    };
    let curse = from.data().color == CardColor::Curse;
    let mut neow = None;
    let rng = match which {
        TransformRng::Neow => neow.insert(run.neow_rng.take().unwrap_or_else(|| StsRandom::new(run.config.seed))),
        TransformRng::Misc => &mut run.rng.misc,
    };
    let k = if curse { rng.random_range(0, list.len() as i32 - 1) } else { rng.random(list.len() as i32 - 1) };
    if let Some(r) = neow {
        run.neow_rng = Some(r);
    }
    list[k as usize]
}

impl NeowOption {
    /// The option's button text in the game (English, formatting codes removed), for checking
    /// rolls against recorded games. `max_hp` and `hp` are the player's when Neow was opened.
    pub fn label(&self, max_hp: i32, hp: i32) -> String {
        use NeowReward::*;
        let bonus = (max_hp as f32 * 0.1) as i32;
        let drawback = match self.drawback {
            NeowDrawback::None => String::new(),
            NeowDrawback::TenPercentHpLoss => format!("[ Lose {bonus} Max HP "),
            NeowDrawback::NoGold => "[ Lose all Gold ".into(),
            NeowDrawback::Curse => "[ Obtain a Curse ".into(),
            NeowDrawback::PercentDamage => format!("[ Take {} damage ", hp / 10 * 3),
        };
        let reward = match self.reward {
            ThreeCards => "[ Choose a Card to obtain ]".into(),
            OneRandomRareCard => "[ Obtain a random rare Card ]".into(),
            RemoveCard => "[ Remove a Card from your deck ]".into(),
            UpgradeCard => "[ Upgrade a Card ]".into(),
            TransformCard => "[ Transform a Card ]".into(),
            RandomColorless => "[ Choose a colorless Card to obtain ]".into(),
            ThreeSmallPotions => "[ Obtain 3 random Potions ]".into(),
            RandomCommonRelic => "[ Obtain a random common Relic ]".into(),
            TenPercentHpBonus => format!("[ Max HP +{bonus} ]"),
            ThreeEnemyKill => "[ Enemies in your next three combats have 1 HP ]".into(),
            HundredGold => "[ Obtain 100 Gold ]".into(),
            RemoveTwo => "Remove 2 Cards ]".into(),
            OneRareRelic => "Obtain a random rare Relic ]".into(),
            ThreeRareCards => "Choose a rare Card to obtain ]".into(),
            TwoFiftyGold => "Gain 250 Gold ]".into(),
            TransformTwoCards => "Transform 2 Cards ]".into(),
            TwentyPercentHpBonus => format!("Max HP +{} ]", bonus * 2),
            RandomColorless2 => "Choose a rare colorless Card to obtain ]".into(),
            BossRelic => "[ Lose your starting Relic Obtain a random boss Relic ]".into(),
        };
        drawback + &reward
    }
}

/// Collapses whitespace so labels compare regardless of spacing.
pub fn normalize_label(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
