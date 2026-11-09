// Tooltip text for the viewer: what cards, relics, potions, powers, intents, orbs and stances do,
// in short hand-written words (no game text is committed or served), filled in with the engine's
// numbers (`info` from crates/sts-wasm, and the view's own numbers in a fight).
import { CARD_TEXT } from "./text-cards.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (n, cls = "") => `<b class="num ${cls}">${n}</b>`;

// ---- Relics (by game id). {N}: the relic's counter where it has one. ----
export const RELIC_TEXT = {
  "Akabeko": "Your first attack each fight deals 8 more damage.",
  "Anchor": "Start each fight with 10 Block.",
  "Ancient Tea Set": "After resting at a rest site, start the next fight with 2 more energy.",
  "Art of War": "If you play no Attacks in a turn, gain 1 more energy next turn.",
  "Astrolabe": "When picked up, transform 3 cards and upgrade them.",
  "Bag of Marbles": "At the start of each fight, apply 1 Vulnerable to every enemy.",
  "Bag of Preparation": "At the start of each fight, draw 2 more cards.",
  "Bird Faced Urn": "Whenever you play a Power, heal 2 HP.",
  "Black Blood": "At the end of each fight, heal 12 HP.",
  "Black Star": "Elites drop 2 relics.",
  "Blood Vial": "At the start of each fight, heal 2 HP.",
  "Bloody Idol": "Whenever you gain gold, heal 5 HP.",
  "Blue Candle": "Curses can be played: lose 1 HP and exhaust them.",
  "Boot": "Attacks that would deal 4 or less unblocked damage deal 5.",
  "Bottled Flame": "An Attack of your choice starts every fight in your hand.",
  "Bottled Lightning": "A Skill of your choice starts every fight in your hand.",
  "Bottled Tornado": "A Power of your choice starts every fight in your hand.",
  "Brimstone": "At the start of your turn, gain 2 Strength; every enemy gains 1.",
  "Bronze Scales": "Start each fight with 3 Thorns.",
  "Burning Blood": "At the end of each fight, heal 6 HP.",
  "Busted Crown": "Gain 1 energy each turn. Card rewards offer 2 fewer cards.",
  "Cables": "Your rightmost orb triggers its passive an extra time.",
  "Calipers": "At the start of your turn, lose only 15 Block instead of all of it.",
  "Calling Bell": "When picked up, gain a curse and 3 relics.",
  "CaptainsWheel": "At the start of turn 3, gain 18 Block.",
  "Cauldron": "When picked up, gain 5 random potions.",
  "Centennial Puzzle": "The first time you lose HP each fight, draw 3 cards.",
  "CeramicFish": "Whenever you add a card to your deck, gain 9 gold.",
  "Champion Belt": "Whenever you apply Vulnerable, also apply 1 Weak.",
  "Charon's Ashes": "Whenever a card is exhausted, hit every enemy for 3.",
  "Chemical X": "X-cost cards get 2 more.",
  "Circlet": "Does nothing (every relic is taken).",
  "CloakClasp": "At the end of your turn, gain 1 Block for each card in your hand.",
  "ClockworkSouvenir": "Start each fight with 1 Artifact.",
  "Coffee Dripper": "Gain 1 energy each turn. You can't rest at rest sites.",
  "Cracked Core": "At the start of each fight, channel 1 Lightning.",
  "CultistMask": "Does nothing in a fight (you caw at the start of each one).",
  "Cursed Key": "Gain 1 energy each turn. Opening a chest gives a curse.",
  "Damaru": "At the start of your turn, gain 1 Mantra.",
  "Darkstone Periapt": "Whenever you gain a curse, gain 6 max HP.",
  "DataDisk": "Start each fight with 1 Focus.",
  "Dead Branch": "Whenever a card is exhausted, add a random card to your hand.",
  "DollysMirror": "When picked up, copy a card in your deck.",
  "Dream Catcher": "Whenever you rest, you may add a card to your deck.",
  "Du-Vu Doll": "Start each fight with 1 Strength for each curse in your deck.",
  "Ectoplasm": "Gain 1 energy each turn. You can't gain gold.",
  "Emotion Chip": "If you lost HP last turn, trigger every orb's passive at the start of this turn.",
  "Empty Cage": "When picked up, remove 2 cards from your deck.",
  "Enchiridion": "At the start of each fight, add a random Power to your hand; it costs 0 this turn.",
  "Eternal Feather": "Entering a rest site heals 3 HP for every 5 cards in your deck.",
  "FaceOfCleric": "Gain 1 max HP after each fight.",
  "FossilizedHelix": "The first time you would lose HP each fight, prevent it (1 Buffer).",
  "Frozen Egg 2": "Powers you add to your deck are upgraded.",
  "Frozen Eye": "You can see the order of your draw pile.",
  "FrozenCore": "If you end your turn with an empty orb slot, channel 1 Frost.",
  "Fusion Hammer": "Gain 1 energy each turn. You can't upgrade cards at rest sites.",
  "Gambling Chip": "At the start of each fight, discard any cards and draw that many.",
  "Ginger": "You can't become Weak.",
  "Girya": "At rest sites you can Lift for 1 Strength each fight (3 times at most). Lifts: {N}.",
  "Golden Idol": "Gain 25% more gold from fights.",
  "GoldenEye": "Whenever you Scry, Scry 2 more.",
  "Gremlin Horn": "Whenever an enemy dies, gain 1 energy and draw 1 card.",
  "GremlinMask": "Start each fight with 1 Weak.",
  "HandDrill": "Whenever you break an enemy's Block, apply 2 Vulnerable.",
  "Happy Flower": "Every 3 turns, gain 1 energy. Count: {N}.",
  "HolyWater": "At the start of each fight, add 3 Miracles to your hand.",
  "HornCleat": "At the start of turn 2, gain 14 Block.",
  "HoveringKite": "The first time you discard a card each turn, gain 1 energy.",
  "Ice Cream": "Unspent energy carries over to the next turn.",
  "Incense Burner": "Every 6 turns, gain 1 Intangible. Count: {N}.",
  "InkBottle": "Every 10 cards played, draw 1 card. Count: {N}.",
  "Inserter": "Every 2 turns, gain 1 orb slot. Count: {N}.",
  "Juzu Bracelet": "Unknown rooms never hold a regular fight.",
  "Kunai": "Every 3 Attacks played in a turn, gain 1 Dexterity.",
  "Lantern": "Gain 1 energy on the first turn of each fight.",
  "Lee's Waffle": "When picked up, gain 7 max HP and heal to full.",
  "Letter Opener": "Every 3 Skills played in a turn, hit every enemy for 5.",
  "Lizard Tail": "The first time you would die, heal to half your max HP instead.",
  "Magic Flower": "Healing in a fight is 50% stronger.",
  "Mango": "When picked up, gain 14 max HP.",
  "Mark of Pain": "Gain 1 energy each turn. Start each fight with 2 Wounds in your draw pile.",
  "Mark of the Bloom": "You can't heal.",
  "Matryoshka": "The next 2 chests hold 2 relics. Left: {N}.",
  "MawBank": "Gain 12 gold whenever you climb a floor, until you spend gold at a shop.",
  "MealTicket": "Entering a shop heals 15 HP.",
  "Meat on the Bone": "If you end a fight at 50% HP or less, heal 12 HP.",
  "Medical Kit": "Status cards can be played: exhaust them.",
  "Melange": "Whenever you shuffle your draw pile, Scry 3.",
  "Membership Card": "Shop prices are halved.",
  "Mercury Hourglass": "At the start of your turn, hit every enemy for 3.",
  "Molten Egg 2": "Attacks you add to your deck are upgraded.",
  "Mummified Hand": "Whenever you play a Power, a random card in your hand costs 0 this turn.",
  "MutagenicStrength": "Start each fight with 3 Strength that goes away at the end of turn 1.",
  "Necronomicon": "The first Attack costing 2 or more each turn is played twice. Comes with a curse.",
  "NeowsBlessing": "The next 3 fights' enemies start at 1 HP. Left: {N}.",
  "Nilry's Codex": "At the end of each turn, you may add 1 of 3 random cards to your draw pile.",
  "Ninja Scroll": "At the start of each fight, add 3 Shivs to your hand.",
  "Nloth's Gift": "Rare cards show up in card rewards more often.",
  "NlothsMask": "The next chest you open holds no relic.",
  "Nuclear Battery": "At the start of each fight, channel 1 Plasma.",
  "Nunchaku": "Every 10 Attacks played, gain 1 energy. Count: {N}.",
  "Odd Mushroom": "You take only 25% more damage from Vulnerable, not 50%.",
  "Oddly Smooth Stone": "Start each fight with 1 Dexterity.",
  "Old Coin": "When picked up, gain 300 gold.",
  "Omamori": "Blocks the next 2 curses you would gain. Left: {N}.",
  "OrangePellets": "Playing a Power, an Attack and a Skill in one turn removes your debuffs.",
  "Orichalcum": "If you end your turn with no Block, gain 6 Block.",
  "Ornamental Fan": "Every 3 Attacks played in a turn, gain 4 Block.",
  "Orrery": "When picked up, add 5 cards to your deck (one from each of 5 card rewards).",
  "Pandora's Box": "When picked up, transform every Strike and Defend.",
  "Pantograph": "At the start of boss fights, heal 25 HP.",
  "Paper Crane": "Weak enemies deal 40% less damage, not 25%.",
  "Paper Frog": "Vulnerable enemies take 75% more damage, not 50%.",
  "Peace Pipe": "At rest sites you can Toke: remove a card from your deck.",
  "Pear": "When picked up, gain 10 max HP.",
  "Pen Nib": "Every 10th Attack deals double damage. Count: {N}.",
  "Philosopher's Stone": "Gain 1 energy each turn. Every enemy starts with 1 Strength.",
  "Pocketwatch": "If you play 3 or fewer cards in a turn, draw 3 more cards next turn.",
  "Potion Belt": "Gain 2 potion slots.",
  "Prayer Wheel": "Regular fights give 2 card rewards.",
  "PreservedInsect": "Elites start with 25% less HP.",
  "PrismaticShard": "Card rewards can hold cards of any colour.",
  "PureWater": "At the start of each fight, add a Miracle to your hand.",
  "Question Card": "Card rewards offer 1 more card.",
  "Red Circlet": "Does nothing (every relic is taken).",
  "Red Mask": "At the start of each fight, apply 1 Weak to every enemy.",
  "Red Skull": "While at 50% HP or less, you have 3 more Strength.",
  "Regal Pillow": "Resting heals 15 more HP.",
  "Ring of the Serpent": "At the start of your turn, draw 1 more card.",
  "Ring of the Snake": "At the start of each fight, draw 2 more cards.",
  "Runic Capacitor": "Start each fight with 3 more orb slots.",
  "Runic Cube": "Whenever you lose HP, draw 1 card.",
  "Runic Dome": "Gain 1 energy each turn. You can't see enemy intents.",
  "Runic Pyramid": "You no longer discard your hand at the end of your turn.",
  "SacredBark": "Potions are twice as strong.",
  "Self Forming Clay": "Whenever you lose HP, gain 3 Block next turn.",
  "Shovel": "At rest sites you can Dig for a relic.",
  "Shuriken": "Every 3 Attacks played in a turn, gain 1 Strength.",
  "Singing Bowl": "On a card reward you may take 2 max HP instead.",
  "SlaversCollar": "In elite and boss fights, gain 1 energy each turn.",
  "Sling": "In elite fights, start with 2 Strength.",
  "Smiling Mask": "Card removal at shops always costs 50 gold.",
  "Snake Skull": "Whenever you apply Poison, apply 1 more.",
  "Snecko Eye": "Draw 2 more cards each turn. Cards you draw get a random cost from 0 to 3.",
  "Sozu": "Gain 1 energy each turn. You can't get potions.",
  "Spirit Poop": "Does nothing.",
  "SsserpentHead": "Entering an unknown room gives 50 gold.",
  "StoneCalendar": "At the end of turn 7, hit every enemy for 52.",
  "Strange Spoon": "Cards that would exhaust have a 50% chance to go to your discard pile instead.",
  "Strawberry": "When picked up, gain 7 max HP.",
  "StrikeDummy": "Cards with \"Strike\" in their name deal 3 more damage.",
  "Sundial": "Every 3 shuffles of your draw pile, gain 2 energy. Count: {N}.",
  "Symbiotic Virus": "At the start of each fight, channel 1 Dark.",
  "TeardropLocket": "Start each fight in Calm.",
  "The Courier": "The shop restocks what you buy. Prices are 20% lower.",
  "The Specimen": "When an enemy dies, its Poison moves to a random enemy.",
  "TheAbacus": "Whenever you shuffle your draw pile, gain 6 Block.",
  "Thread and Needle": "Start each fight with 4 Plated Armor.",
  "Tingsha": "Whenever you discard a card during your turn, hit a random enemy for 3.",
  "Tiny Chest": "Every 4th unknown room is a treasure room. Count: {N}.",
  "Tiny House": "When picked up: 1 potion, 50 gold, 5 max HP, 1 card and an upgrade.",
  "Toolbox": "At the start of each fight, add 1 of 3 random colourless cards to your hand.",
  "Torii": "Unblocked attack damage of 5 or less is reduced to 1.",
  "Tough Bandages": "Whenever you discard a card during your turn, gain 3 Block.",
  "Toxic Egg 2": "Skills you add to your deck are upgraded.",
  "Toy Ornithopter": "Whenever you drink a potion, heal 5 HP.",
  "TungstenRod": "Whenever you would lose HP, lose 1 less.",
  "Turnip": "You can't become Frail.",
  "TwistedFunnel": "At the start of each fight, apply 4 Poison to every enemy.",
  "Unceasing Top": "Whenever your hand is empty during your turn, draw 1 card.",
  "Vajra": "Start each fight with 1 Strength.",
  "Velvet Choker": "Gain 1 energy each turn. You can't play more than 6 cards a turn.",
  "VioletLotus": "Leaving Calm gives 1 more energy.",
  "War Paint": "When picked up, upgrade 2 random Skills.",
  "WarpedTongs": "At the start of your turn, upgrade a random card in your hand for this fight.",
  "Whetstone": "When picked up, upgrade 2 random Attacks.",
  "White Beast Statue": "Fights always drop a potion.",
  "WingedGreaves": "On the map you may fly to any room on the next floor (3 times). Left: {N}.",
  "WristBlade": "Attacks that cost 0 deal 4 more damage.",
  "Yang": "Whenever you play a Power, gain 1 Dexterity this turn (Duality).",
};

// ---- Potions (by game id). {P}: the potency (doubled with Sacred Bark). ----
export const POTION_TEXT = {
  "Ambrosia": "Enter Divinity.",
  "Ancient Potion": "Gain {P} Artifact.",
  "AttackPotion": "Pick 1 of 3 random Attacks to add to your hand; it costs 0 this turn.",
  "BlessingOfTheForge": "Upgrade every card in your hand for this fight.",
  "Block Potion": "Gain {P} Block.",
  "BloodPotion": "Heal {P}% of your max HP.",
  "BottledMiracle": "Add {P} Miracles to your hand.",
  "ColorlessPotion": "Pick 1 of 3 random colourless cards to add to your hand; it costs 0 this turn.",
  "CultistPotion": "Gain {P} Ritual (Strength at the end of each turn).",
  "CunningPotion": "Add {P} Shiv+ to your hand.",
  "Dexterity Potion": "Gain {P} Dexterity.",
  "DistilledChaos": "Play the top {P} cards of your draw pile.",
  "DuplicationPotion": "Your next card is played twice.",
  "ElixirPotion": "Exhaust any number of cards in your hand.",
  "Energy Potion": "Gain {P} energy.",
  "EntropicBrew": "Fill every empty potion slot with random potions.",
  "EssenceOfDarkness": "Channel {P} Dark for each orb slot.",
  "EssenceOfSteel": "Gain {P} Plated Armor.",
  "Explosive Potion": "Hit every enemy for {P}.",
  "FairyPotion": "When you would die, heal to {P}% of your max HP instead (used up).",
  "FearPotion": "Apply {P} Vulnerable.",
  "Fire Potion": "Hit an enemy for {P}.",
  "FocusPotion": "Gain {P} Focus.",
  "Fruit Juice": "Gain {P} max HP.",
  "GamblersBrew": "Discard any number of cards, then draw that many.",
  "GhostInAJar": "Gain {P} Intangible.",
  "HeartOfIron": "Gain {P} Metallicize.",
  "LiquidBronze": "Gain {P} Thorns.",
  "LiquidMemories": "Put {P} card from your discard pile into your hand; it costs 0 this turn.",
  "Poison Potion": "Apply {P} Poison.",
  "PotionOfCapacity": "Gain {P} orb slots.",
  "PowerPotion": "Pick 1 of 3 random Powers to add to your hand; it costs 0 this turn.",
  "Regen Potion": "Gain {P} Regeneration.",
  "SkillPotion": "Pick 1 of 3 random Skills to add to your hand; it costs 0 this turn.",
  "SmokeBomb": "Escape a non-boss fight (no rewards).",
  "SneckoOil": "Draw {P} cards. Give the cards in your hand random costs.",
  "SpeedPotion": "Gain {P} Dexterity this turn.",
  "StancePotion": "Enter Calm or Wrath.",
  "SteroidPotion": "Gain {P} Strength this turn.",
  "Strength Potion": "Gain {P} Strength.",
  "Swift Potion": "Draw {P} cards.",
  "Weak Potion": "Apply {P} Weak.",
};

// ---- Powers (by game id). {A}: the amount. ----
export const POWER_TEXT = {
  "Strength": "Attacks deal {A} more damage.",
  "Dexterity": "Block from cards is {A} more.",
  "Vulnerable": "Takes 50% more attack damage, for {A} turns.",
  "Weakened": "Deals 25% less attack damage, for {A} turns.",
  "Frail": "Gains 25% less Block from cards, for {A} turns.",
  "Artifact": "Blocks the next {A} debuffs.",
  "Metallicize": "At the end of the turn, gain {A} Block.",
  "Plated Armor": "At the end of the turn, gain {A} Block. Unblocked attack damage removes 1.",
  "Poison": "Loses {A} HP at the start of its turn, then Poison drops by 1.",
  "Barricade": "Block doesn't go away at the start of the turn.",
  "Anger": "Gains {A} Strength whenever you play a Skill.",
  "Angry": "Gains {A} Strength whenever it takes attack damage.",
  "Berserk": "Gain {A} energy at the start of each turn.",
  "Brutality": "At the start of your turn, lose 1 HP and draw {A} card(s).",
  "Combust": "At the end of your turn, lose 1 HP and hit every enemy for {A}.",
  "Corruption": "Skills cost 0 and exhaust when played.",
  "Dark Embrace": "Whenever a card is exhausted, draw {A} card(s).",
  "Demon Form": "Gain {A} Strength at the start of each turn.",
  "Double Tap": "The next {A} Attack(s) this turn are played twice.",
  "Entangled": "Can't play Attacks this turn.",
  "Evolve": "Whenever you draw a Status card, draw {A} card(s).",
  "Feel No Pain": "Whenever a card is exhausted, gain {A} Block.",
  "Fire Breathing": "Whenever you draw a Status or Curse, hit every enemy for {A}.",
  "Flame Barrier": "Attackers take {A} damage back this turn.",
  "Juggernaut": "Whenever you gain Block, hit a random enemy for {A}.",
  "Flex": "Lose {A} Strength at the end of the turn.",
  "No Draw": "Can't draw more cards this turn.",
  "Rage": "Gain {A} Block whenever you play an Attack this turn.",
  "Rupture": "Gain {A} Strength whenever you lose HP from a card.",
  "Shackled": "Gets back {A} Strength at the end of the turn.",
  "Vigor": "The next Attack deals {A} more damage.",
  "Thorns": "Attackers take {A} damage back.",
  "Pen Nib": "The next Attack deals double damage.",
  "Buffer": "Prevents the next {A} time(s) you would lose HP.",
  "IntangiblePlayer": "Damage and HP loss are reduced to 1, for {A} turn(s).",
  "Intangible": "Damage and HP loss are reduced to 1, for {A} turn(s).",
  "Next Turn Block": "Gain {A} Block next turn.",
  "Confusion": "Drawn cards get a random cost from 0 to 3.",
  "Regeneration": "Heal {A} HP at the end of the turn, then it drops by 1.",
  "DuplicationPower": "The next {A} card(s) are played twice.",
  "DexLoss": "Lose {A} Dexterity at the end of each turn.",
  "Sadistic": "Whenever you apply a debuff, the enemy takes {A} damage.",
  "NoBlockPower": "Can't gain Block from cards for {A} turn(s).",
  "TheBomb": "Explodes in a few turns for {A} damage to every enemy.",
  "Panache": "Every fifth card in a turn hits every enemy for {A}.",
  "Mayhem": "Plays the top {A} card(s) of the draw pile at the start of the turn.",
  "Magnetism": "Adds {A} random colourless card(s) to your hand each turn.",
  "Curl Up": "The first time it takes attack damage, gains {A} Block.",
  "Mode Shift": "Changes to defensive mode after taking {A} more damage.",
  "Ritual": "Gains {A} Strength at the end of each turn.",
  "Sharp Hide": "Whenever you play an Attack, you take {A} damage.",
  "Split": "Splits into two smaller slimes at half HP.",
  "Spore Cloud": "When it dies, applies {A} Vulnerable to you.",
  "Thievery": "Steals {A} gold with each attack.",
  "Minion": "Leaves when its leader dies; no rewards for killing it.",
  "Hex": "Whenever you play a non-Attack, shuffle {A} Dazed into your draw pile.",
  "Flight": "Takes half attack damage; loses 1 Flight per hit, falling at 0.",
  "Malleable": "Gains {A} Block when hit, growing with each hit this turn.",
  "Painful Stabs": "Its unblocked attacks shuffle a Wound into your discard pile.",
  "Stasis": "Holds one of your cards; you get it back when it dies.",
  "Explosive": "Explodes in {A} turn(s).",
  "Fading": "Dies in {A} turn(s).",
  "Shifting": "Loses Strength equal to the unblocked damage it takes this turn.",
  "Generic Strength Up Power": "Gains {A} Strength at the end of each turn.",
  "Constricted": "Takes {A} damage at the end of the turn.",
  "Compulsive": "Gains {A} Block at the end of its turn.",
  "Slow": "Takes 10% more damage for each card played this turn ({A} so far).",
  "Life Link": "Revives while another of its kind lives.",
  "Time Warp": "After 12 cards played, ends your turn and gains 2 Strength ({A} played).",
  "Draw Reduction": "Draw 1 card fewer next turn.",
  "Regenerate": "Heals {A} HP at the end of its turn.",
  "Curiosity": "Gains {A} Strength whenever you play a Power.",
  "Unawakened": "Rises again with full HP when killed the first time.",
  "Surrounded": "Attacks from behind deal 50% more damage; face the enemy you attack.",
  "BackAttack": "Deals 50% more damage while behind you.",
  "Invincible": "Can lose at most {A} more HP this turn.",
  "BeatOfDeath": "Whenever you play a card, you take {A} damage.",
  "Thousand Cuts": "Whenever you play a card, hit every enemy for {A}.",
  "Accuracy": "Shivs deal {A} more damage.",
  "After Image": "Whenever you play a card, gain {A} Block.",
  "Blur": "Block isn't removed at the start of the next {A} turn(s).",
  "Burst": "The next {A} Skill(s) this turn are played twice.",
  "Choked": "Loses {A} HP whenever you play a card this turn.",
  "CorpseExplosionPower": "When it dies, every enemy takes damage equal to its max HP.",
  "Envenom": "Unblocked attack damage applies {A} Poison.",
  "Energized": "Gain {A} energy next turn.",
  "EnergizedBlue": "Gain {A} energy next turn.",
  "Infinite Blades": "Add {A} Shiv(s) to your hand at the start of each turn.",
  "Noxious Fumes": "Apply {A} Poison to every enemy at the start of each turn.",
  "Phantasmal": "Attacks deal double damage next turn.",
  "Double Damage": "Attacks deal double damage this turn.",
  "Draw Card": "Draw {A} more card(s) next turn.",
  "Tools Of The Trade": "At the start of the turn, draw {A} and discard {A} card(s).",
  "Retain Cards": "Keep up to {A} card(s) in hand at the end of the turn.",
  "Wraith Form v2": "Lose {A} Dexterity at the end of each turn.",
  "Night Terror": "Next turn, copies of a card come to your hand.",
  "Focus": "Orbs are {A} stronger.",
  "Lockon": "Orbs deal 50% more damage to it, for {A} turn(s).",
  "Electro": "Lightning hits every enemy.",
  "Loop": "Your next orb triggers its passive {A} more time(s) each turn.",
  "Storm": "Whenever you play a Power, channel {A} Lightning.",
  "StaticDischarge": "Whenever you take attack damage, channel {A} Lightning.",
  "Heatsink": "Whenever you play a Power, draw {A} card(s).",
  "Hello": "Add {A} random Common card(s) to your hand each turn.",
  "Creative AI": "Add {A} random Power(s) to your hand each turn.",
  "Echo Form": "The first {A} card(s) each turn are played twice.",
  "Bias": "Lose {A} Focus at the start of each turn.",
  "Draw": "Draw {A} more card(s) each turn.",
  "Repair": "Heal {A} HP at the end of the fight.",
  "Amplify": "The next {A} Power(s) this turn are played twice.",
  "Equilibrium": "Keep your hand at the end of the turn, for {A} turn(s).",
  "Rebound": "The next card played goes on top of your draw pile.",
  "Mantra": "At 10 Mantra, enter Divinity ({A} now).",
  "BattleHymn": "Add {A} Smite(s) to your hand each turn.",
  "BlockReturnPower": "Attacking it gives you {A} Block.",
  "Controlled": "Mind Bloom's effect.",
  "DevaForm": "Gain {A} energy at the start of the turn, growing by 1 each turn.",
  "DevotionPower": "Gain {A} Mantra at the start of each turn.",
  "EndTurnDeath": "You die at the start of your next turn.",
  "EnergyDownPower": "Gain {A} less energy each turn.",
  "EstablishmentPower": "Retained cards cost {A} less.",
  "FreeAttackPower": "The next Attack costs 0.",
  "LikeWaterPower": "At the end of the turn, if in Calm, gain {A} Block.",
  "MasterRealityPower": "Cards created in a fight are upgraded.",
  "Nirvana": "Whenever you Scry, gain {A} Block.",
  "OmegaPower": "At the end of the turn, hit every enemy for {A}.",
  "PathToVictoryPower": "Marked: loses HP equal to its Mark ({A}) when Mark is triggered.",
  "Study": "Shuffle {A} Insight(s) into your draw pile at the end of the turn.",
  "WaveOfTheHandPower": "Whenever you gain Block this turn, apply {A} Weak to every enemy.",
  "WireheadingPower": "Scry {A} at the start of the turn.",
  "WrathNextTurnPower": "Enter Wrath next turn and draw {A} card(s).",
  "Adaptation": "Whenever you enter Wrath, draw {A} card(s).",
  "Collect": "Add a Miracle+ to your hand next turn, for {A} turn(s).",
};

// ---- Intents ----
export const INTENT_ICON = {
  Attack: "⚔", AttackBuff: "⚔", AttackDebuff: "⚔", AttackDefend: "⚔",
  Buff: "▲", Debuff: "▼", StrongDebuff: "▼", Defend: "🛡", DefendDebuff: "🛡", DefendBuff: "🛡",
  Escape: "⤴", Magic: "✶", None: "·", Sleep: "z", Stun: "✕", Unknown: "?", Debug: "?",
};
export const INTENT_TEXT = {
  Attack: "Means to attack",
  AttackBuff: "Means to attack and grow stronger",
  AttackDebuff: "Means to attack and weaken you",
  AttackDefend: "Means to attack and block",
  Buff: "Means to grow stronger",
  Debuff: "Means to weaken you",
  StrongDebuff: "Means to badly weaken you",
  Defend: "Means to block",
  DefendDebuff: "Means to block and weaken you",
  DefendBuff: "Means to block and grow stronger",
  Escape: "Means to run away",
  Magic: "Means to do something unusual",
  None: "Does nothing",
  Sleep: "Asleep",
  Stun: "Stunned",
  Unknown: "Its intent is hidden",
  Debug: "Its intent is unknown",
};

export const ORB_TEXT = {
  Lightning: "Each turn: hit a random enemy for {P}. Evoke: hit a random enemy for {E}.",
  Frost: "Each turn: gain {P} Block. Evoke: gain {E} Block.",
  Dark: "Each turn: its damage grows by {P} (now {E}). Evoke: hit the enemy with the least HP for {E}.",
  Plasma: "Each turn: gain {P} energy. Evoke: gain {E} energy.",
  Empty: "An empty orb slot.",
};

export const STANCE_TEXT = {
  Calm: "Leaving Calm gives 2 energy.",
  Wrath: "You deal and take double attack damage.",
  Divinity: "You deal triple attack damage. Gives 3 energy on entering; leaves at the start of your next turn.",
  Neutral: "No stance.",
};

export const ROOM_TEXT = {
  Monster: "Fight", Elite: "Elite fight", Event: "Unknown", Rest: "Rest site", Shop: "Shop",
  Treasure: "Treasure", Boss: "Boss",
};

// ---- Filling in ----

let INFO = { cards: {}, relics: {}, potions: {}, monsters: {} };
const POTION_BY_NAME = {};
export function setInfo(info) {
  INFO = info;
  for (const [id, p] of Object.entries(info.potions)) POTION_BY_NAME[p.name] = id;
}
export const info = () => INFO;

/// A card's numbers: those shown in a fight when given (`c.damage` etc. from the view), else the
/// card's own (`c.base_damage` etc. from the deck and the piles: Searing Blow upgraded several
/// times, a grown Ritual Dagger), else the fresh or upgraded copy's from `info`.
export function cardStats(c) {
  const d = INFO.cards[c.id] || {};
  const base = (c.upgraded ? d.up : d) || {};
  const own = (k) => (c[`base_${k}`] !== undefined ? c[`base_${k}`] : base[k]);
  return {
    type: c.card_type || d.type || "",
    rarity: c.rarity || d.rarity || "",
    color: d.color || "",
    cost: c.cost !== undefined ? c.cost : base.cost,
    damage: c.inFight && d.type === "Attack" ? c.damage : own("damage"),
    block: c.inFight && c.block > 0 ? c.block : own("block"),
    magic: c.inFight ? c.magic : own("magic"),
    baseDamage: base.damage,
    baseBlock: base.block,
    exhaust: base.exhaust,
    ethereal: base.ethereal,
    innate: base.innate,
    retain: base.retain,
  };
}

function fill(t, upgraded, values) {
  t = t.replace(/\[([^|\]]*)\|([^\]]*)\]/g, (_, a, b) => (upgraded ? a : b));
  return esc(t).replace(/\{(\w)\}/g, (_, k) => {
    const v = values[k];
    if (!v) return "?";
    return num(v.n, v.cls);
  });
}

/// A card's text as HTML.
export function cardText(c) {
  const s = cardStats(c);
  const t = CARD_TEXT[c.id];
  const cmp = (n, base) => (n > base ? "up" : n < base ? "down" : "");
  let html = t === undefined ? "" : fill(t, c.upgraded, {
    D: { n: s.damage, cls: cmp(s.damage, s.baseDamage) },
    B: { n: s.block, cls: cmp(s.block, s.baseBlock) },
    M: { n: s.magic },
  });
  const kw = [];
  if (s.cost === -2 && s.type !== "Power") kw.push("Unplayable");
  if (s.innate) kw.push("Innate");
  if (s.retain) kw.push("Retain");
  if (s.ethereal) kw.push("Ethereal");
  if (s.exhaust) kw.push("Exhaust");
  if (kw.length) html += ` <span class="kw">${kw.join(". ")}.</span>`;
  return html.trim();
}

export function relicText(id, counter) {
  const t = RELIC_TEXT[id];
  if (!t) return "";
  return esc(t).replace("{N}", counter !== undefined && counter >= 0 ? num(counter) : "–");
}

/// A potion by game id or by name; `bark`: Sacred Bark doubles it.
export function potionText(idOrName, bark = false) {
  const id = INFO.potions[idOrName] ? idOrName : POTION_BY_NAME[idOrName];
  const t = POTION_TEXT[id];
  if (!t) return "";
  const p = (INFO.potions[id] || {}).potency || 0;
  return esc(t).replace("{P}", num(bark ? p * 2 : p));
}
export const potionId = (idOrName) => (INFO.potions[idOrName] ? idOrName : POTION_BY_NAME[idOrName]);

export function powerText(id, amount) {
  const t = POWER_TEXT[id];
  if (!t) return "";
  return esc(t).replace(/\{A\}/g, num(amount)).replace(/card\(s\)/g, amount === 1 ? "card" : "cards").replace(/turn\(s\)/g, amount === 1 ? "turn" : "turns")
    .replace(/time\(s\)/g, amount === 1 ? "time" : "times").replace(/\((s)\)/g, amount === 1 ? "" : "s");
}

export function orbText(o) {
  const t = ORB_TEXT[o.id];
  return t ? esc(t).replace(/\{P\}/g, num(o.passive)).replace(/\{E\}/g, num(o.evoke)) : "";
}

// ---- Keywords, explained in a card's tooltip when its text mentions them ----
export const KEYWORDS = {
  Block: "Stops that much damage until the start of your next turn.",
  Vulnerable: "Takes 50% more attack damage.",
  Weak: "Deals 25% less attack damage.",
  Frail: "Gains 25% less Block from cards.",
  Strength: "Adds to attack damage.",
  Dexterity: "Adds to Block from cards.",
  Exhaust: "Removed from play until the fight ends.",
  Ethereal: "Exhausted if it is still in your hand at the end of your turn.",
  Retain: "Stays in your hand at the end of your turn.",
  Innate: "Starts each fight in your hand.",
  Unplayable: "Can't be played.",
  Poison: "Loses that much HP at the start of its turn; then Poison drops by 1.",
  Shiv: "A 0-cost Attack that hits for 4 and exhausts.",
  Channel: "Put an orb in your next slot; with every slot full, the oldest is evoked first.",
  Evoke: "Use up an orb for its stronger effect.",
  Lightning: "An orb: hits a random enemy each turn; evoked, hits harder.",
  Frost: "An orb: gives Block each turn; evoked, gives more.",
  Dark: "An orb: its damage grows each turn; evoked, hits the weakest enemy.",
  Plasma: "An orb: gives 1 energy each turn; evoked, gives 2.",
  Focus: "Makes orbs stronger.",
  Scry: "Look at that many cards on top of your draw pile and discard any of them.",
  Mantra: "At 10 Mantra, enter Divinity.",
  Calm: "A stance: leaving it gives 2 energy.",
  Wrath: "A stance: you deal and take double attack damage.",
  Divinity: "A stance: you deal triple attack damage; it gives 3 energy and lasts until your next turn.",
  Intangible: "Damage and HP loss are reduced to 1.",
  Artifact: "Blocks the next debuff.",
  "Plated Armor": "Gain that much Block at the end of each turn; unblocked attack damage removes 1.",
  Thorns: "Attackers take that much damage back.",
  Metallicize: "Gain that much Block at the end of each turn.",
  "Lock-On": "Orbs deal 50% more damage to it.",
  Mark: "Mark makes an enemy lose HP when it is triggered.",
  Upgrade: "Upgraded cards are stronger (marked with +).",
  Miracle: "A 0-cost Skill that gives 1 energy, Retains and exhausts.",
  Smite: "A 1-cost Attack that hits for 12, Retains and exhausts.",
  Insight: "A 0-cost Skill that draws 2 cards, Retains and exhausts.",
  Wound: "An unplayable status card.",
  Dazed: "An unplayable status card that exhausts at the end of the turn.",
  Burn: "An unplayable status card that hurts you at the end of the turn.",
  Void: "An unplayable status card that costs you 1 energy when drawn.",
};

/// The keywords a piece of text mentions, with what they mean.
export function keywordsIn(html) {
  const text = html.replace(/<[^>]+>/g, "");
  return Object.entries(KEYWORDS).filter(([k]) => new RegExp(`\\b${k}s?\\b`).test(text));
}
