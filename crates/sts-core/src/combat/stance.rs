//! The Watcher's stances (`AbstractStance` and its four kinds) and `ChangeStanceAction`.

use serde::{Deserialize, Serialize};

use super::Combat;
use crate::action::{Action, Cr, DamageType};
use crate::cards::CardId;
use crate::powers::PowerId;
use crate::relics::RelicId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stance {
    #[default]
    Neutral,
    Calm,
    Wrath,
    Divinity,
}

impl Stance {
    /// `AbstractStance.ID`.
    pub fn java_id(self) -> &'static str {
        match self {
            Stance::Neutral => "Neutral",
            Stance::Calm => "Calm",
            Stance::Wrath => "Wrath",
            Stance::Divinity => "Divinity",
        }
    }

    pub fn from_java_id(id: &str) -> Option<Stance> {
        [Stance::Neutral, Stance::Calm, Stance::Wrath, Stance::Divinity].into_iter().find(|s| s.java_id() == id)
    }

    /// `atDamageGive`: Wrath doubles and Divinity triples the player's normal damage.
    pub fn at_damage_give(self, dmg: f32, t: DamageType) -> f32 {
        match (self, t) {
            (Stance::Wrath, DamageType::Normal) => dmg * 2.0,
            (Stance::Divinity, DamageType::Normal) => dmg * 3.0,
            _ => dmg,
        }
    }

    /// `atDamageReceive`: Wrath doubles the normal damage the player takes.
    pub fn at_damage_receive(self, dmg: f32, t: DamageType) -> f32 {
        match (self, t) {
            (Stance::Wrath, DamageType::Normal) => dmg * 2.0,
            _ => dmg,
        }
    }
}

impl Combat {
    /// `ChangeStanceAction.update`.
    pub(crate) fn change_stance(&mut self, new: Stance) {
        let old = self.player.stance;
        if old != new {
            // The powers' onChangeStance, in list order.
            for (id, amount) in self.player.powers.iter().map(|p| (p.id, p.amount)).collect::<Vec<_>>() {
                match id {
                    PowerId::MentalFortress => self.push_bot(Action::GainBlock { target: Cr::Player, amount }),
                    PowerId::Rushdown if new == Stance::Wrath => {
                        self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false })
                    }
                    _ => {}
                }
            }
            // The relics' onChangeStance: Violet Lotus.
            if old == Stance::Calm && self.has_relic(RelicId::VioletLotus) {
                self.push_bot(Action::GainEnergy { amount: 1 });
            }
            // onExitStance, then onEnterStance.
            if old == Stance::Calm {
                self.push_bot(Action::GainEnergy { amount: 2 });
            }
            self.player.stance = new;
            if new == Stance::Divinity {
                self.push_bot(Action::GainEnergy { amount: 3 });
            }
            // The discard pile's triggerExhaustedCardsOnStanceChange: Flurry of Blows comes back.
            let flurries: Vec<u32> = self.discard_pile.iter().filter(|c| c.id == CardId::FlurryOfBlows).map(|c| c.instance).collect();
            for instance in flurries {
                self.push_bot(Action::DiscardToHandCard { instance });
            }
        }
        self.on_modify_power();
    }

    /// The stance's `atStartOfTurn`: Divinity lasts one turn.
    pub(crate) fn stance_at_start_of_turn(&mut self) {
        if self.player.stance == Stance::Divinity {
            self.push_bot(Action::ChangeStance { stance: Stance::Neutral });
        }
    }
}
