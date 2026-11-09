//! Game actions (`AbstractGameAction` subclasses), collapsed to single-step effects.

use serde::{Deserialize, Serialize};

use crate::cards::Card;
use crate::monsters::{EnemyMove, Monster};
use crate::powers::{Power, PowerId};

/// Argument to `AbstractMonster.changeState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonsterState {
    Closed,
    Open,
    Rear,
    RearIdle,
    Debuff,
    Activate,
    ActivateOrb,
    Deactivate,
    DefensiveMode,
    OffensiveMode,
    ResetThreshold,
    UseNet,
    /// Shelled Parasite's Plated Armor is gone.
    ArmorBreak,
    /// Byrd's Flight is gone.
    Grounded,
    Flying,
    /// Darkling coming back from half-dead.
    Revive,
    /// Awakened One's second form.
    Rebirth,
}

/// A creature reference: the player or an index into the monster group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cr {
    Player,
    Monster(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DamageType {
    Normal,
    Thorns,
    HpLoss,
}

/// A resolved `DamageInfo`: who dealt it, how much, and of which type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DamageInfo {
    pub owner: Option<Cr>,
    pub output: i32,
    pub damage_type: DamageType,
}

impl DamageInfo {
    pub fn new(owner: Option<Cr>, output: i32, damage_type: DamageType) -> DamageInfo {
        DamageInfo { owner, output, damage_type }
    }
}

/// Where a `DamageAction` reads its numbers from. Monster attacks share a `DamageInfo` object
/// with the monster whose `output` is refreshed by `applyPowers`, so they are read at execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DamageSource {
    Fixed(DamageInfo),
    MonsterSlot { monster: usize, slot: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Action {
    Damage { target: Cr, source: DamageSource, steal_gold: i32 },
    DamageAllEnemies { source: Option<Cr>, damage: Vec<i32>, damage_type: DamageType },
    VampireDamageAllEnemies { damage: Vec<i32> },
    DamageRandomEnemy { info: DamageInfo },
    PummelDamage { target: usize, info: DamageInfo },
    Feed { target: usize, info: DamageInfo, max_hp: i32 },
    /// `RitualDaggerAction`: a kill grows every copy of the card (and the master deck's).
    RitualDagger { target: usize, info: DamageInfo, amount: i32, uuid: u32 },
    LoseHp { target: Cr, source: Option<Cr>, amount: i32 },
    Heal { target: Cr, amount: i32 },
    /// `IncreaseMaxHpAction` on a monster.
    IncreaseMaxHp { target: usize, amount: i32 },
    GainBlock { target: Cr, amount: i32 },
    LoseBlock { target: Cr, amount: i32 },
    DoubleYourBlock { target: Cr },
    GainBlockRandomMonster { source: usize, amount: i32 },
    ApplyPower { target: Cr, source: Option<Cr>, power: Power, amount: i32 },
    ReducePower { target: Cr, id: PowerId, amount: i32 },
    RemovePower { target: Cr, id: PowerId },
    GainEnergy { amount: i32 },
    /// `LoseEnergyAction` (Void).
    LoseEnergy { amount: i32 },
    GainEnergyAndEnableControls { amount: i32 },
    DrawCard { amount: i32, follow_up: Option<Box<Action>>, shuffle_check: bool },
    /// `DrawCardAction(n, followUp, clearDrawHistory = false)`: the rest of a draw that had to
    /// shuffle, adding to the same `drawnCards`.
    DrawCardKeepHistory { amount: i32, follow_up: Option<Box<Action>> },
    EmptyDeckShuffle,
    Exhaust { amount: i32, random: bool, any_number: bool, can_pick_zero: bool },
    ExhaustSpecificInHand { instance: u32 },
    ExhaustAllEthereal,
    ExhaustAllNonAttack,
    Discard { amount: i32, random: bool, end_turn: bool },
    DiscardAtEndOfTurn,
    RestoreRetainedCards,
    ClearCardQueue,
    MakeTempCardInDiscard { card: Card, amount: i32 },
    MakeTempCardInHand { card: Card, amount: i32 },
    MakeTempCardInDrawPile { card: Card, amount: i32, random_spot: bool, to_bottom: bool },
    UseCard { card: Card, target: Option<usize>, exhaust_card: bool },
    /// `NewQueueCardAction()` with no card: queues the end-of-turn marker.
    QueueEndTurn,
    /// `NewQueueCardAction(card, ...)` holding the card (the card is no longer in any pile).
    QueueCard { card: Card, target: Option<usize>, immediate: bool, autoplay: bool, #[serde(default)] random_target: bool },
    PlayTopCard { target: Option<usize>, exhaust: bool },
    /// The anonymous action queued at the end of `AbstractRoom.endTurn()`.
    EndTurnSequence,
    EndTurn,
    MonsterStartTurn,
    RollMove { monster: usize },
    SetMove { monster: usize, mv: EnemyMove },
    ChangeState { monster: usize, state: MonsterState },
    Escape { monster: usize },
    Suicide { monster: usize, trigger_relics: bool },
    SpawnMonster { monster: Box<Monster>, minion: bool },
    CannotLose,
    CanLose,
    /// Looter's and Mugger's inline action that records how much gold the next hit will steal.
    LooterRecordGold { monster: usize },
    /// `VampireDamageAction`: damage, then the attacker heals the HP the target lost.
    VampireDamage { target: Cr, source: DamageSource },
    /// `SpawnMonsterAction(m, isMinion)` for a monster already built into `monsters` (so its
    /// summoner can track it) but not yet in the group.
    SpawnPrepared { monster: usize, minion: bool },
    /// `SummonGremlinAction` for a gremlin already built into `monsters`.
    SummonGremlin { monster: usize },
    /// `ApplyStasisAction` (Bronze Orb).
    ApplyStasis { monster: usize },
    /// `MakeTempCardInHandAction(card, false, true)`: the card itself comes back.
    MakeTempCardInHandSame { card: Card },
    /// `MakeTempCardInDiscardAction(card, true)`.
    MakeTempCardInDiscardSame { card: Card },
    /// `MakeTempCardInDiscardAndDeckAction`: one copy shuffled into the draw pile, one discarded.
    MakeTempCardInDiscardAndDeck { card: Card },
    /// `AddCardToDeckAction`: the card joins the master deck (Writhing Mass).
    AddCardToDeck { card: crate::cards::CardId },
    // Card-specific actions.
    Whirlwind { damage: Vec<i32>, free_to_play_once: bool, energy_on_use: i32 },
    SpotWeakness { amount: i32, target: Option<usize> },
    PutOnDeck { amount: i32, random: bool },
    ModifyDamage { uuid: u32, amount: i32 },
    LimitBreak,
    FiendFire { target: usize, info: DamageInfo },
    Exhume { upgrade: bool },
    DualWield { amount: i32 },
    Dropkick { target: usize, info: DamageInfo },
    DiscardPileToTopOfDeck,
    BurnIncrease,
    BlockPerNonAttack { amount: i32 },
    AttackDamageRandomEnemy { card: Card },
    Armaments { upgraded: bool },
    // Relic actions.
    /// Toolbox: pick one of three random colorless cards.
    ChooseOneColorless,
    /// Gambling Chip: discard any number of cards, then draw that many.
    GamblingChip,
    /// Red Skull's inline check at battle start.
    RedSkullCheck,
    /// `RemoveDebuffsAction` (Orange Pellets, Champ).
    RemoveDebuffs { target: Cr },
    // Potion actions.
    /// `DiscoveryAction`: pick one of three generated cards (`amount` copies, cost 0 this turn).
    Discovery { card_type: Option<crate::cards::CardType>, colorless: bool, amount: i32 },
    /// `BetterDiscardPileToHandAction(amount, 0)` (Liquid Memories, `set_cost`) or
    /// `BetterDiscardPileToHandAction(amount)` (Hologram).
    DiscardToHand { amount: i32, set_cost: bool },
    /// `RandomizeHandCostAction` (Snecko Oil).
    RandomizeHandCost,
    /// `ObtainPotionAction(returnRandomPotion(true))` (Entropic Brew).
    ObtainRandomPotion,
    /// `RegenAction`: heal, then the player's Regeneration drops by one.
    Regen { target: Cr },
    // Colorless card actions.
    /// `ConditionalDrawAction(n, ATTACK)` (Impatience).
    ConditionalDraw { amount: i32 },
    /// `SkillFromDeckToHandAction` / `AttackFromDeckToHandAction` (Secret Technique/Weapon).
    FromDeckToHand { card_type: crate::cards::CardType },
    /// `ShuffleAction(drawPile, false)` (Deep Breath).
    ShuffleDrawPile,
    /// `DrawPileToHandAction(n, type)` (Violence).
    DrawPileToHand { amount: i32, card_type: crate::cards::CardType },
    Apotheosis,
    /// `GreedAction` (Hand of Greed).
    Greed { target: usize, info: DamageInfo, gold: i32 },
    Forethought { any: bool },
    Enlightenment { combat: bool },
    Transmutation { upgraded: bool, free: bool, energy_on_use: i32 },
    Madness,
    /// The Bomb's countdown for one bomb (`uid`).
    ReduceBomb { uid: u32 },
    /// Mayhem's inline action: picks a target, then queues `PlayTopCardAction`.
    MayhemPlay,
    /// `UpgradeRandomCardAction` (Warped Tongs).
    UpgradeRandomCard,
    /// `CodexAction` (Nilry's Codex).
    Codex,
    // The Silent.
    /// `PoisonLoseHpAction`: the poison's damage, then it counts down.
    PoisonLoseHp { target: Cr, amount: i32 },
    /// `DiscardSpecificCardAction` on a hand card.
    DiscardSpecific { instance: u32 },
    Bane { target: usize, info: DamageInfo },
    /// `BouncingFlaskAction`: poison `amount` on `target`, then bounce `times - 1` more times.
    BouncingFlask { target: Option<usize>, amount: i32, times: i32 },
    /// `ApplyBulletTimeAction`: the hand costs 0 this turn.
    BulletTime,
    CalculatedGamble,
    /// `DoublePoisonAction` (`extra` 1) and `TriplePoisonAction` (`extra` 2).
    MultiplyPoison { target: usize, extra: i32 },
    Doppelganger { upgraded: bool, free: bool, energy_on_use: i32 },
    /// `EscapePlanAction`: block if the draw that queued it drew a Skill.
    EscapePlan { block: i32 },
    Expertise { amount: i32 },
    /// `DamagePerAttackPlayedAction` (Finisher).
    DamagePerAttackPlayed { target: usize, info: DamageInfo },
    /// `FlechetteAction`: a hit per Skill in hand.
    Flechette { target: usize, info: DamageInfo },
    HeelHook { target: usize, info: DamageInfo },
    Malaise { target: usize, upgraded: bool, free: bool, energy_on_use: i32 },
    /// `NightmareAction`: pick a hand card; `amount` copies of it come next turn.
    Nightmare { amount: i32 },
    /// `SetupAction`: a hand card goes on top of the draw pile, free once.
    Setup,
    Skewer { target: usize, damage: i32, free: bool, energy_on_use: i32 },
    GainEnergyIfDiscard { amount: i32 },
    /// `BladeFuryAction` (Storm of Steel): discard the hand, a Shiv for each card.
    BladeFury { upgraded: bool },
    /// `UnloadAction`: discard every non-Attack in hand.
    Unload,
    /// `RetainCardsAction` (Well-Laid Plans): keep up to `amount` hand cards.
    RetainCards { amount: i32 },
    /// `ApplyPowerToRandomEnemyAction` with poison (The Specimen).
    PoisonRandomEnemy { amount: i32 },
    // The Defect.
    /// `ChannelAction`; with `auto_evoke` false it only fills an empty slot (Recursion).
    Channel { orb: crate::combat::Orb, auto_evoke: bool },
    EvokeOrb { count: i32 },
    EvokeAllOrbs,
    EvokeWithoutRemovingOrb { count: i32 },
    RemoveAllOrbs,
    IncreaseMaxOrb { amount: i32 },
    DecreaseMaxOrb { amount: i32 },
    TriggerEndOfTurnOrbs,
    /// `LightningOrbEvokeAction` / `LightningOrbPassiveAction`.
    LightningOrbDamage { amount: i32, hit_all: bool },
    /// `DarkOrbEvokeAction`: its target was picked when it was made.
    DarkOrbEvoke { target: Option<usize>, amount: i32 },
    /// `ImpulseAction` (Emotion Chip) or `DarkImpulseAction` (Darkness+, `dark_only`).
    Impulse { dark_only: bool },
    EssenceOfDarkness { amount: i32 },
    AggregateEnergy { divide: i32 },
    /// `AllCostToHandAction(0)` (All For One).
    AllCostToHand,
    /// `DiscardToHandAction(card)`.
    DiscardToHandCard { instance: u32 },
    Barrage { target: usize, info: DamageInfo },
    CompileDriver { amount: i32 },
    DoubleEnergy,
    Ftl { target: usize, info: DamageInfo, count: i32 },
    Fission { upgraded: bool },
    /// `ForTheEyesAction` (Go for the Eyes).
    ForTheEyes { target: usize, amount: i32 },
    /// `GashAction` (Claw): every Claw's damage grows, the one in play too.
    Gash { amount: i32 },
    /// `IncreaseMiscAction` (Genetic Algorithm): its block grows, the master deck's card too.
    IncreaseMisc { uuid: u32, amount: i32 },
    Multicast { upgraded: bool, free: bool, energy_on_use: i32 },
    Recycle,
    /// `RedoAction` (Recursion).
    Redo,
    ReinforcedBody { block: i32, free: bool, energy_on_use: i32 },
    /// `ScrapeFollowUpAction`: discard the drawn cards that cost something.
    ScrapeFollowUp,
    /// `ShuffleAllAction` (Reboot).
    ShuffleAll,
    Sunder { target: usize, info: DamageInfo },
    Tempest { upgraded: bool, free: bool, energy_on_use: i32 },
    /// `BetterDrawPileToHandAction` (Seek).
    DrawPileToHandChoice { amount: i32 },
    RemoveAllBlock { target: Cr },
    /// `ModifyBlockAction` (Steam Barrier).
    ModifyBlock { uuid: u32, amount: i32 },
    /// `ReduceCostAction(uuid, n)` (Streamline): every copy costs 1 less this combat.
    ReduceCost { uuid: u32 },
    // The Watcher.
    /// `ChangeStanceAction`.
    ChangeStance { stance: crate::combat::Stance },
    /// `ScryAction` (`amount` includes Golden Eye's +2, added when the action is made).
    Scry { amount: i32 },
    /// `PressEndTurnButtonAction` (Conclude, Meditate, Vault).
    PressEndTurnButton,
    /// `SkipEnemiesTurnAction` (Vault).
    SkipEnemiesTurn,
    /// `ChooseOneAction` (Wish, Stance Potion): one of the option cards' `onChoseThisOption`.
    ChooseOne { options: Vec<Card> },
    ForeignInfluence { upgraded: bool },
    /// `MeditateAction`: cards from the discard pile return to the hand, retained.
    Meditate { amount: i32 },
    /// `OmniscienceAction`: a draw-pile card is played `amount` times.
    Omniscience { amount: i32 },
    Collect { upgraded: bool, free: bool, energy_on_use: i32 },
    /// `ConjureBladeAction` (`energy_on_use` already has Conjure Blade+'s +1).
    ConjureBlade { free: bool, energy_on_use: i32 },
    /// Follow-Up, Crush Joints, Sash Whip (`HeadStompAction`), Sanctity: `then` (added to the
    /// top) if the card played before this one was of `card_type`.
    PreviousCardCheck { card_type: crate::cards::CardType, then: Box<Action> },
    FearNoEvil { target: usize, info: DamageInfo },
    /// `HaltAction`: `extra` more block in Wrath.
    Halt { block: i32, extra: i32 },
    Indignation { amount: i32 },
    InnerPeace { amount: i32 },
    Judgement { target: usize, cutoff: i32 },
    LessonLearned { target: usize, info: DamageInfo },
    Wallop { target: usize, info: DamageInfo },
    /// `TriggerMarksAction` for Pressure Points.
    TriggerMarks,
    /// `EstablishmentPowerAction`: retained cards cost less for the combat.
    Establishment { amount: i32 },
    /// `GainGoldAction` (Fame and Fortune).
    GainGold { amount: i32 },
    /// `ReduceCostAction(card)`: that one card costs 1 less this combat (Sands of Time).
    ReduceCostCard { instance: u32 },
}

impl Action {
    /// `GameActionManager.clearPostCombatActions` keeps heals, block, card uses and DAMAGE-typed actions.
    pub fn survives_combat_end(&self) -> bool {
        matches!(
            self,
            Action::Heal { .. }
                | Action::GainBlock { .. }
                | Action::UseCard { .. }
                | Action::Damage { .. }
                | Action::DamageAllEnemies { .. }
                | Action::VampireDamageAllEnemies { .. }
                | Action::DamageRandomEnemy { .. }
                | Action::PummelDamage { .. }
                | Action::Feed { .. }
                | Action::RitualDagger { .. }
                | Action::VampireDamage { .. }
                | Action::LoseHp { .. }
                | Action::Suicide { .. }
                | Action::PoisonLoseHp { .. }
                | Action::Bane { .. }
                | Action::DamagePerAttackPlayed { .. }
                | Action::DarkOrbEvoke { .. }
                | Action::Sunder { .. }
                | Action::LessonLearned { .. }
                | Action::Wallop { .. }
        )
    }
}
