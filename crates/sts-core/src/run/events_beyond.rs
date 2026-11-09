//! Act 3 events (`com.megacrit.cardcrawl.events.beyond`).

use sts_rng::{java_shuffle, JavaRandom};

use super::{Event, EventId};
use crate::cards::{Card, CardId, CardRarity, CardType};
use crate::relics::{RelicId, RelicTier};
use crate::run::{Reward, Run, Screen};

impl Run {
    /// `AbstractDungeon.getColorlessRewardCards()`.
    fn colorless_reward_cards(&mut self) -> Vec<Card> {
        let mut n = 3;
        if self.has_relic(RelicId::QuestionCard) {
            n += 1;
        }
        if self.has_relic(RelicId::BustedCrown) {
            n -= 2;
        }
        let mut ids: Vec<CardId> = Vec::new();
        for _ in 0..n {
            let rarity = if self.rng.card.random_boolean_chance(0.3) { CardRarity::Rare } else { CardRarity::Uncommon };
            if rarity == CardRarity::Rare {
                self.card_blizz = 5;
            }
            let draw = |run: &mut Run| run.pools.random_colorless(rarity, &mut run.rng.card).or_else(|| run.pools.random_colorless(CardRarity::Uncommon, &mut run.rng.card));
            let mut card = draw(self);
            while card.is_some_and(|c| ids.contains(&c)) {
                card = draw(self);
            }
            if let Some(c) = card {
                ids.push(c);
            }
        }
        ids.into_iter()
            .map(|id| {
                let mut c = self.new_card(id);
                self.preview_obtain(&mut c);
                c
            })
            .collect()
    }

    /// `CardHelper.returnCardOfType(type, miscRng)` over the unbottled deck, as a deck index.
    fn random_card_of_type(&mut self, t: CardType) -> i32 {
        let cands: Vec<usize> = (0..self.deck.len()).filter(|&i| !self.deck[i].bottled && self.deck[i].card_type() == t).collect();
        if cands.is_empty() {
            return -1;
        }
        cands[self.rng.misc.random(cands.len() as i32 - 1) as usize] as i32
    }

    /// The constructors of the Act 3 events.
    pub(super) fn open_beyond_event(&mut self, e: &mut Event) {
        use EventId::*;
        let asc15 = self.config.ascension >= 15;
        match e.id {
            Falling => {
                // n: skill, power, attack (deck indices, -1 when there is none).
                let attack = self.random_card_of_type(CardType::Attack);
                let skill = self.random_card_of_type(CardType::Skill);
                let power = self.random_card_of_type(CardType::Power);
                e.n = [skill, power, attack, 0];
                e.set(&[("continue", true)]);
            }
            MindBloom => {
                let rich = self.floor % 50 <= 40;
                e.n[0] = rich as i32;
                let third = if rich { "I am Rich: 999 gold, 2 Normality" } else { "I am Healthy: full heal, Doubt" };
                e.set(&[("I am War: fight an Act 1 boss", true), ("I am Awake: upgrade all, Mark of the Bloom", true), (third, true)]);
            }
            MoaiHead => {
                e.n[0] = crate::run::map::java_round(self.max_hp as f32 * if asc15 { 0.18 } else { 0.125 });
                let idol = self.has_relic(RelicId::GoldenIdol);
                e.set(&[("jump in: lose max HP, heal fully", true), ("offer Golden Idol: 333 gold", idol), ("leave", true)]);
            }
            MysteriousSphere => {
                self.roll_event_monsters("2 Orb Walkers");
                e.set(&[("open the sphere: fight", true), ("leave", true)]);
            }
            SecretPortal => e.set(&[("enter the portal: go to the boss", true), ("leave", true)]),
            SensoryStone => e.set(&[("continue", true)]),
            TombRedMask => {
                if self.has_relic(RelicId::RedMask) {
                    e.set(&[("don the mask: 222 gold", true), ("leave", true)]);
                } else {
                    e.set(&[("don the mask", false), ("pay all gold: Red Mask", true), ("leave", true)]);
                }
            }
            WindingHalls => {
                e.n[0] = crate::run::map::java_round(self.max_hp as f32 * if asc15 { 0.18 } else { 0.125 });
                e.n[1] = crate::run::map::java_round(self.max_hp as f32 * if asc15 { 0.2 } else { 0.25 });
                e.n[2] = crate::run::map::java_round(self.max_hp as f32 * 0.05);
                e.set(&[("continue", true)]);
            }
            _ => {}
        }
    }

    /// Picks option `k` of an Act 3 event.
    pub(super) fn beyond_event_choose(&mut self, mut e: Event, k: usize) -> Result<(), String> {
        use EventId::*;
        match (e.id, e.screen) {
            (Falling, 0) => {
                e.screen = 1;
                if e.n[..3].iter().all(|&i| i < 0) {
                    e.set(&[("continue", true)]);
                } else {
                    let name = |run: &Run, i: i32| if i >= 0 { run.deck[i as usize].name() } else { String::new() };
                    let skill = format!("lose {}", name(self, e.n[0]));
                    let power = format!("lose {}", name(self, e.n[1]));
                    let attack = format!("lose {}", name(self, e.n[2]));
                    e.set(&[(&skill, e.n[0] >= 0), (&power, e.n[1] >= 0), (&attack, e.n[2] >= 0)]);
                }
            }
            (Falling, _) => {
                let i = e.n[k.min(2)];
                if i >= 0 {
                    self.take_cards(&[i as usize]);
                    self.on_master_deck_change();
                }
                e.leave(2);
            }
            (MindBloom, _) => match k {
                0 => {
                    let mut bosses = vec!["The Guardian", "Hexaghost", "Slime Boss"];
                    let seed = self.rng.misc.random_long();
                    java_shuffle(&mut bosses, &mut JavaRandom::new(seed));
                    let gold = if self.config.ascension >= 13 { 25 } else { 50 };
                    let relic = self.random_relic(RelicTier::Rare);
                    self.event_combat(bosses[0], vec![Reward::Gold(gold), Reward::Relic(relic)], false);
                    return Ok(());
                }
                1 => {
                    for c in self.deck.iter_mut() {
                        if c.can_upgrade() {
                            c.upgrade();
                        }
                    }
                    self.obtain_relic(RelicId::MarkOfTheBloom);
                    e.leave(1);
                }
                _ => {
                    if e.n[0] == 1 {
                        self.gain_gold(999);
                        for _ in 0..2 {
                            let c = self.new_card(CardId::Normality);
                            self.obtain_card(c);
                        }
                    } else {
                        let m = self.max_hp;
                        self.heal(m);
                        let c = self.new_card(CardId::Doubt);
                        self.obtain_card(c);
                    }
                    e.leave(1);
                }
            },
            (MoaiHead, _) => {
                match k {
                    0 => {
                        self.max_hp -= e.n[0];
                        self.hp = self.hp.min(self.max_hp);
                        self.max_hp = self.max_hp.max(1);
                        let m = self.max_hp;
                        self.heal(m);
                    }
                    1 => {
                        self.lose_relic(RelicId::GoldenIdol);
                        self.gain_gold(333);
                    }
                    _ => {}
                }
                e.leave(1);
            }
            (MysteriousSphere, 0) => {
                if k == 0 {
                    e.screen = 1;
                    e.set(&[("fight", true)]);
                } else {
                    e.leave(2);
                }
            }
            (MysteriousSphere, _) => {
                let gold = self.rng.misc.random_range(45, 55);
                let relic = self.random_screenless_relic(RelicTier::Rare);
                self.event_combat("2 Orb Walkers", vec![Reward::Gold(gold), Reward::Relic(relic)], false);
                return Ok(());
            }
            (SecretPortal, 0) => {
                if k == 0 {
                    e.screen = 1;
                    e.set(&[("continue", true)]);
                } else {
                    e.leave(2);
                }
            }
            (SecretPortal, _) => {
                // A boss room node at (-1, 15), entered straight from the event.
                let x = self.pos.map_or(0, |(x, _)| x);
                self.pos = Some((x, crate::run::map::MAP_HEIGHT as i32 - 1));
                self.enter_node(x);
                return Ok(());
            }
            (SensoryStone, 0) => {
                e.screen = 1;
                e.set(&[("recall: 1 colorless card", true), ("recall: 2 colorless cards, lose 5 HP", true), ("recall: 3 colorless cards, lose 10 HP", true)]);
            }
            (SensoryStone, _) => {
                // getRandomMemory shuffles the memory texts with miscRng.
                let mut memories = [0u8, 1, 2, 3];
                let seed = self.rng.misc.random_long();
                java_shuffle(&mut memories, &mut JavaRandom::new(seed));
                let n = k + 1;
                let items: Vec<Reward> = (0..n).map(|_| Reward::Cards(self.colorless_reward_cards())).collect();
                if k > 0 {
                    self.event_damage(5 * k as i32, false, true);
                    if matches!(self.screen, Screen::Over) {
                        return Ok(());
                    }
                }
                self.event_rewards(items);
                return Ok(());
            }
            (TombRedMask, _) => {
                let mask = self.has_relic(RelicId::RedMask);
                if k == 0 {
                    self.gain_gold(222);
                } else if k == 1 && !mask {
                    self.event_lose_gold(self.gold);
                    self.obtain_relic(RelicId::RedMask);
                } else {
                    self.screen = Screen::Map;
                    return Ok(());
                }
                e.leave(1);
            }
            (WindingHalls, 0) => {
                e.screen = 1;
                let madness = format!("embrace madness: lose {} HP, 2 Madness", e.n[0]);
                let writhe = format!("retrace: heal {}, Writhe", e.n[1]);
                let max = format!("press on: lose {} max HP", e.n[2]);
                e.set(&[(&madness, true), (&writhe, true), (&max, true)]);
            }
            (WindingHalls, _) => {
                match k {
                    0 => {
                        self.event_damage(e.n[0], false, false);
                        if matches!(self.screen, Screen::Over) {
                            return Ok(());
                        }
                        for _ in 0..2 {
                            let c = self.new_card(CardId::Madness);
                            self.obtain_card(c);
                        }
                    }
                    1 => {
                        self.heal(e.n[1]);
                        let c = self.new_card(CardId::Writhe);
                        self.obtain_card(c);
                    }
                    _ => self.decrease_max_hp(e.n[2]),
                }
                e.leave(2);
            }
            _ => return Err(format!("{:?} screen {} is not handled", e.id, e.screen)),
        }
        if !matches!(self.screen, Screen::Over) {
            self.show(e);
        }
        Ok(())
    }
}
