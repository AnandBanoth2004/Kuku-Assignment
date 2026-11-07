"""Cue lexicons + a per-element feature extractor for the OFFLINE (rule-based) engine.

HONEST SCOPE: these are hand-written genre cues (melodrama / romance / revenge microdrama), combined with
*structural* signals (questions, withheld replies, interrupted lines, sudden entrances).  They are a cheap
proxy for semantic understanding - the LLM layer (src/llm_client.py) exists precisely because a lexicon cannot
read context.  All weights were set by hand and eyeballed on the supplied 50 episodes (no held-out set), so
they are engineering assumptions, not measured truths.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .models import Element


def _rx(words: Iterable[str]) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % "|".join(words), re.I)


# ---- shock / stakes vocabulary in speech (weight per hit; summed then saturated) -------------------------
_STRONG = _rx([r"kill\w*", r"murder\w*", r"die[sd]?", r"dead", r"death", r"blood\w*", r"hang\w*", r"gallows", r"execut\w+",
               r"burn\w*", r"poison\w*", r"arsenic", r"torture\w*", r"brand\w*", r"traitor\w*", r"treason\w*",
               r"betray\w*", r"alive", r"you are not", r"rightful", r"imposter", r"impostor", r"how dare you",
               r"take her", r"seize (?:her|him|them)", r"never forgive", r"make you pay", r"you will pay",
               r"witch\w*", r"dungeon", r"chains?", r"scream\w*", r"explo\w+", r"your life"])
_MEDIUM = _rx([r"lie[sd]?", r"liar", r"lying", r"stole\w*", r"steal\w*", r"swore", r"abandon\w*", r"humiliat\w+", r"disgrace\w*",
               r"worthless", r"disown\w*", r"banish\w*", r"cast out", r"get out", r"guards?", r"arrest\w*", r"prison\w*",
               r"gaol", r"trial", r"verdict", r"sentenc\w+", r"secret\w*", r"truth", r"reveal\w*", r"conspir\w+",
               r"changed my mind", r"never loved", r"marry\w*|marrying", r"wedding", r"husband", r"wife", r"bride",
               r"groom", r"ring", r"accept my hand", r"mine", r"regret", r"unless", r"last chance", r"or else",
               r"or (?:watch|I)", r"tomorrow", r"forbidden", r"duke", r"duchess", r"crown", r"throne", r"obey",
               r"command\w*", r"shame\w*", r"unworthy", r"beggar", r"peasant", r"powerless", r"mercy", r"beg\w*",
               r"don't touch", r"take your hands off", r"let (?:her|him|me) go", r"stand aside", r"who are you",
               r"how long", r"enough"])
_WEAK = _rx([r"love", r"heart", r"promise", r"never", r"always", r"forever", r"family", r"father", r"mother", r"sister",
             r"brother", r"please", r"alone", r"afraid", r"fear\w*", r"pain\w*", r"tears?", r"cry\w*"])

# ---- delivery tags: max over tokens -----------------------------------------------------------------
_DELIVERY = {
    1.0: ["scream", "shout", "furious", "fury", "enrag", "outrag", "livid", "hysteric", "roar", "sob", "wail", "agoniz",
          "anguish", "desperat", "beg", "terrif", "panick", "venom", "vicious", "murder", "maniac", "snarl", "shriek",
          "devastat", "heartbr", "horrif", "menac", "threat", "deadly", "chilling", "ominous", "unhinged", "frantic",
          "choking", "gasping", "screeching", "raging", "seething"],
    0.65: ["angry", "bitter", "cruel", "mock", "scorn", "smug", "gloat", "accus", "defian", "trembl", "tearful", "shock",
           "stunn", "urgent", "alarm", "stern", "firm", "command", "fierce", "grim", "intense", "cold", "icy", "haughty",
           "indignant", "disgust", "pained", "betray", "steely", "resolute", "taunt", "sneer", "vengeful", "ruthless",
           "imperious", "authoritative", "incredulous", "panting", "fearful", "wounded", "broken",
           "deris", "contempt", "gleeful", "sinister", "spiteful", "vindictive", "callous", "disdain", "malic", "sadist",
           "scheming", "calculating", "cunning", "wicked", "triumphant", "dismissive", "arrogant", "boast", "pompous",
           "condescend", "patroniz", "conspirat", "carnal", "derisive", "scoff", "goad", "gloating"],
    0.4: ["whisper", "low", "hush", "quiet", "solemn", "grave", "uneasy", "anxious", "tense", "sharp", "blunt"],
    0.2: ["gentle", "soft", "tender", "warm", "calm", "sincere", "reflective", "peaceful", "loving", "affection", "hopeful",
          "polite", "curious", "amused", "cheerful", "proud", "steady"],
}
_DELIV_RX = [(w, re.compile("|".join(stems), re.I)) for w, stems in _DELIVERY.items()]

# ---- visual / action shock --------------------------------------------------------------------------
_VIS_STRONG = _rx([r"slap\w*", r"strik\w+|struck", r"punch\w*", r"drag\w*", r"seiz\w+", r"grab\w*", r"shov\w+", r"hurl\w*",
                   r"smash\w*", r"crash\w*", r"explo\w+", r"fireball", r"blast\w*", r"burst\w*", r"collaps\w+",
                   r"blood\w*", r"bleed\w*", r"wound\w*", r"inferno", r"flames?", r"blade", r"sabre", r"sword", r"dagger",
                   r"pistol", r"gun", r"poker", r"branding", r"searing", r"noose", r"gallows", r"manacle\w*", r"gasp\w*",
                   r"scream\w*", r"convuls\w+", r"choke\w*", r"drops? (?:onto|to) (?:one|her|his) knees?",
                   r"detonat\w+", r"shatter\w*", r"tear[s]? her grip", r"lunges?"])
_VIS_MED = _rx([r"freez\w+|frozen", r"stunned", r"horrif\w+", r"terrif\w+", r"tears?", r"trembl\w+", r"clench\w*",
                r"glar\w+", r"menac\w+", r"ominous", r"threat\w+", r"stagger\w*", r"stumbl\w+", r"rips?|ripped",
                r"throws?|threw", r"door", r"footsteps", r"knock\w*", r"enters?|entering|emerges?", r"silhouett\w+",
                r"ring", r"letter", r"blade", r"chains?", r"crown", r"whisper\w*", r"slam\w*"])

# ---- structural / semantic patterns -----------------------------------------------------------------
RE_QUESTION_END = re.compile(r"\?[?!]*\s*[\"'\u201d]?\s*$")
RE_ELLIPSIS_END = re.compile(r"(?:\.\.\.|…)\s*[\"'”]?\s*$")
RE_ELLIPSIS_START = re.compile(r"^\s*(?:\.\.\.|…)")
RE_REPLY_START = re.compile(r"^\s*(?:yes|no|because|then|and|but|so|well|exactly|indeed|of course|i see|very well|perhaps|"
                            r"not anymore|that is|that's|it is|it's|he is|she is|they are)\b", re.I)
RE_ADDRESS = re.compile(r"^\s*(?:guards?\b|stop\b|wait\b|listen\b|look\b|get out|leave\b|go\b|let me|let her|let him|take your|"
                        r"don't\b|do not\b|never\b|how dare|you (?:will|dare|are|have|lied|knew|stole|killed|betrayed|think|"
                        r"cannot|can't|should|shall|must|did|let|swore)|who are you|what have you|what did you|where is|where did)", re.I)
RE_DEMAND = re.compile(r"\b(?:tell me|speak\b|answer me|answer\b|choose\b|decide\b|or\b.{0,40}\b(?:hang|die|burn|lose|watch)|unless|"
                       r"last chance|you have until|give me|hand (?:it|that) over|accept my hand|marry me|swear\b|"
                       r"look at me|come with me|say it|prove it|name your|bring me|where (?:is|are|did)|"
                       r"or I will|or watch|or leave)", re.I)
RE_THREAT = re.compile(r"\b(?:gallows|hang\w*|burn\w*|kill\w*|die\b|dead\b|you will (?:pay|regret|hang|die|burn|suffer)|"
                       r"regret|make you|ashes|ash\b|one flame|touches your skin|before he sends you|sends you to|"
                       r"never (?:see|leave|forgive)|you will never|i will destroy|destroy you|crush you|take everything|"
                       r"nothing left|you are finished|ten days|tomorrow|become a witch|stop being a prisoner|settle what)", re.I)
RE_REVEAL = re.compile(r"\b(?:you are not|you're not|it was you|he is alive|she is alive|is alive|i am (?:the|your|his|her)\b|"
                       r"the truth|i know what you|i know who|all along|never told|the real\b|rightful|dead duke|"
                       r"same men|the men behind|mastermind|was behind|behind it|knew all along|i was never|"
                       r"not (?:his|her|your) (?:lawful|wife|husband)|lawful wife|illegitimate|impostor|imposter|"
                       r"unmask\w*|your mother|your father)\b", re.I)
RE_RESOLVED = re.compile(r"\b(?:thank you|thanks|i forgive|i love you|goodbye|farewell|at last|we are safe|you are safe|"
                         r"i accept\b|yes[,.!]? i will|welcome home|peace\b|together now|i don't feel afraid|"
                         r"we begin again|begin again|i was wrong|perhaps i was wrong|you didn't say a word|i didn't need to|"
                         r"sleep well|good night|rest now)\b", re.I)
RE_WARM = re.compile(r"\b(?:warm\w*|peaceful\w*|tender\w*|serene|melancholic|lullaby|embers|hearth|sunlight|contented|"
                     r"comfort\w*|cozy|affection\w*|loving\w*|fading? out into silence|quiet (?:peace|contentment))\b", re.I)
# NOTE: the script's own boilerplate ("cliffhanger", "cut to black") is deliberately NOT a cue: ep 16 labels a tender
# gaze "a dramatic cliffhanger hold", and "cut to black" closes resolved episodes too.
RE_SUDDEN = re.compile(r"\b(?:door (?:bursts?|slams?|latch)|latch (?:turns?|clacks?|clicks?)|knock\w*|footsteps|explo\w+|detonat\w+|"
                       r"fireball|bursts? (?:in|open|through)|barges?|slams?|crash\w*|collaps\w+|seizes?|grabs?|drags?|slaps?|strikes?|"
                       r"struck|stabs?|shoots?|gasps?|screams?|convuls\w+|discovers?|sees|witness\w*|stumbles? (?:upon|onto)|"
                       r"freez\w+|frozen|stunned silence|inches from|millimeters? from|silhouett\w+|emerge\w*|looms?)\b", re.I)
RE_REACTION_HOLD = re.compile(r"\b(?:stunned|frozen|freez\w+|wide[- ]eyed|eyes (?:widen|wide|snap)|shock\w*|horror|horrif\w+|"
                              r"realiz\w+|dawning|jaw drops?|gapes?|petrified|terrified|pales?|stares?|gaz\w+|holds? on|"
                              r"push[- ]in|slow push|dramatic hold|lingers?)\b", re.I)
RE_OBJECT_REVEAL = re.compile(r"\b(?:macro|extreme close-up|insert|tight close-up|close-up)\b[^.]{0,60}\b(?:ring|watch|letter|note|"
                              r"document|seal|signet|locket|key|vial|bottle|blade|dagger|knife|poison|crown|brand|iron|chain|"
                              r"ledger|will|portrait|photograph|scroll|goblet|cup)\b", re.I)
RE_STING = re.compile(r"^\W*(?:her|him|them|you|me|who|what|how|impossible|alive|you're alive|it can't be)\W*[?!]{1,3}\W*$", re.I)
RE_NAMEISH = re.compile(r"\b[A-Z][a-z]{2,}\b")
RE_GREETING = re.compile(r"^\s*(?:good (?:morning|evening|day)|hello|welcome|thank you|please (?:sit|come in)|pardon|excuse me)\b", re.I)


def _saturate(x: float, k: float = 1.4) -> float:
    return 1.0 - math.exp(-x / k)


def shock_text(text: str) -> float:
    """0..1 stakes/shock density of a spoken line (saturating weighted keyword hits)."""
    s = 1.0 * len(_STRONG.findall(text)) + 0.55 * len(_MEDIUM.findall(text)) + 0.2 * len(_WEAK.findall(text))
    return _saturate(s)


def delivery_intensity(tags: Iterable[str]) -> float:
    best = 0.0
    for tag in tags:
        for w, rx in _DELIV_RX:
            if rx.search(tag):
                best = max(best, w)
                break
    return best


def visual_shock(text: str) -> float:
    s = 1.0 * len(_VIS_STRONG.findall(text)) + 0.45 * len(_VIS_MED.findall(text))
    return _saturate(s, 1.2)


@dataclass
class ElFeat:
    intensity: float = 0.0      # 0..1 emotional arousal of the element
    shock: float = 0.0          # 0..1 shock / stakes
    delivery: float = 0.0
    words: int = 0
    question: bool = False
    ellipsis_end: bool = False
    ellipsis_start: bool = False
    reply_start: bool = False
    address: bool = False
    demand: bool = False
    threat: bool = False
    reveal: bool = False
    resolved: bool = False
    warm: bool = False
    sudden: bool = False
    hold: bool = False
    greeting: bool = False
    exclaim: int = 0
    sting: bool = False          # short disbelief reaction ("Her?!")
    object_reveal: bool = False  # insert / macro shot on a story object (ring, watch, letter...)
    strong_vis: bool = False     # at least one STRONG physical/visual shock word (tears / whispers alone do not count)


def analyze_element(el: Element) -> ElFeat:
    t = el.text or ""
    f = ElFeat(words=len(re.findall(r"[A-Za-z']+", t)))
    if el.kind == "dialogue":
        f.shock = shock_text(t)
        f.delivery = delivery_intensity(el.delivery)
        f.exclaim = t.count("!")
        f.question = bool(RE_QUESTION_END.search(t))
        f.ellipsis_end = bool(RE_ELLIPSIS_END.search(t))
        f.ellipsis_start = bool(RE_ELLIPSIS_START.search(t))
        f.reply_start = bool(RE_REPLY_START.match(t))
        f.address = bool(RE_ADDRESS.match(t))
        f.demand = bool(RE_DEMAND.search(t))
        f.threat = bool(RE_THREAT.search(t))
        f.reveal = bool(RE_REVEAL.search(t))
        f.resolved = bool(RE_RESOLVED.search(t))
        f.greeting = bool(RE_GREETING.match(t))
        f.sting = bool(RE_STING.match(t))
        caps = sum(1 for w in re.findall(r"[A-Za-z]{3,}", t) if w.isupper())
        f.intensity = min(1.0, 0.42 * f.delivery + 0.30 * f.shock + 0.08 * min(f.exclaim, 2)
                          + 0.06 * (1 if f.question and f.shock > 0.2 else 0) + 0.06 * min(caps, 1)
                          + 0.08 * (1 if f.threat or f.reveal else 0))
        if el.voice == "V.O.":
            f.intensity *= 0.85
    else:
        v = visual_shock(t)
        f.shock = v
        f.sudden = bool(RE_SUDDEN.search(t))
        f.warm = bool(RE_WARM.search(t))
        f.hold = bool(RE_REACTION_HOLD.search(t))
        f.object_reveal = bool(RE_OBJECT_REVEAL.search(t))
        f.strong_vis = bool(_VIS_STRONG.search(t))
        f.resolved = False
        base = {"action": 0.18, "sfx": 0.12, "camera": 0.08}.get(el.kind, 0.1)
        f.intensity = min(1.0, base + 0.75 * v + (0.08 if f.sudden else 0.0))
    return f
