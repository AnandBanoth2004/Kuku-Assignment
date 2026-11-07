"""A SYNTHETIC screenplay (written for the tests, not from the supplied pack) in the same file layout.

Used to test the reusable new-screenplay workflow, the validator (by mutating a real promo record) and format tolerance.
"""
from __future__ import annotations

SPEC = [  # scene, start, end, [(kind, label, start, end, body)]
    (1, "INT. HARBOR OFFICE - NIGHT", 0.0, 48.0, [
        ("dialogue", "MAYA (Panicked, Breathless)", 1.0, 3.2, "Captain! They found the ledger!"),
        ("dialogue", "CAPTAIN REYES (Cold, Menacing)", 3.6, 6.0, "Who else has seen it?"),
        ("dialogue", "MAYA (Trembling)", 6.6, 9.4, "Only the inspector. He knows everything."),
        ("action", "ACTION", 9.6, 12.4, "Reyes seizes the lantern and smashes it against the wall; glass shatters across the floor."),
        ("sfx", "SFX", 9.6, 12.4, "A violent crash of breaking glass."),
        ("dialogue", "CAPTAIN REYES (Furious, Shouting)", 12.8, 16.0, "Then the inspector will never leave this harbor alive!"),
        ("dialogue", "MAYA (Desperate, Begging)", 16.6, 19.4, "Captain, please... he has a family."),
        ("dialogue", "CAPTAIN REYES (Icy, Threatening)", 19.8, 23.0, "So did the men who trusted you."),
        ("dialogue", "MAYA (Whispering)", 24.0, 27.0, "I will tell them the truth."),
        ("dialogue", "CAPTAIN REYES (Dangerous)", 27.6, 31.0, "Then you will die beside him."),
        ("action", "ACTION", 31.0, 36.0, "He draws a pistol and levels it at Maya's chest, his hand perfectly steady."),
    ]),
    (2, "EXT. HARBOR DOCK - NIGHT", 48.0, 100.0, [
        ("dialogue", "INSPECTOR VOSS (Commanding, Loud)", 50.0, 53.0, "Drop the weapon, Reyes! The harbor guard is outside!"),
        ("dialogue", "CAPTAIN REYES (Shocked)", 53.6, 56.0, "You came here alone?"),
        ("dialogue", "INSPECTOR VOSS (Grim)", 56.6, 58.6, "No."),
        ("sfx", "SFX", 59.0, 62.0, "The door bursts open; boots thunder across the boards."),
        ("dialogue", "MAYA (Gasping)", 62.4, 65.0, "They are here. They are all here."),
        ("dialogue", "CAPTAIN REYES (Furious, Shouting)", 66.0, 69.5, "Who sent you?! Answer me!"),
        ("dialogue", "INSPECTOR VOSS (Cold)", 70.5, 74.5, "The one person you swore was dead."),
        ("action", "ACTION", 75.0, 80.0, "A cloaked figure steps out of the fog, a silver locket gleaming in the lantern light."),
        ("dialogue", "CAPTAIN REYES (Horrified, Whispering)", 80.6, 83.5, "It cannot be you."),
        ("dialogue", "WOMAN IN CLOAK (Calm, Deadly)", 84.5, 88.0, "Hello, Captain. I have been waiting for this."),
    ]),
]


def build_text(number: int = 1, spec=SPEC, upper_meta: bool = False, crlf: bool = False, bom: bool = False) -> str:
    L = [f"EPISODE NUMBER: Episode {number}", "EPISODE LOGLINE: A smuggler captain is exposed by a ledger and cornered by a figure he thought dead.",
         "CONTINUITY NOTES: Standalone synthetic test episode.", "", "--- SCRIPT START ---", ""]
    for sc, heading, s0, s1, els in spec:
        sh, ss, se, aad = ("SCENE HEADING:", "SCENE START:", "SCENE END:", "ACTIONS AND DIALOGUE:") if upper_meta else ("Scene heading:", "Scene start:", "Scene end:", "Actions and dialogue:")
        L += [f"SCENE {sc:02d}", f"{sh} {heading}", f"{ss} {_t(s0)}", f"{se} {_t(s1)}", "", aad, ""]
        cams, t = [], s0
        while t < s1 - 0.01:
            e = min(round(t + 3.5, 2), s1)
            cams.append(("camera", "CAMERA", t, e, "Medium shot holding on the speaker." if int(t) % 2 else "Close-up on the listener's face."))
            t = e
        for kind, label, a, b, body in sorted(els + cams, key=lambda x: (x[2], x[0] != "camera")):
            L += [f"**{label}:**", f"Start: ({_t(a)}) | End: ({_t(b)})", f'"{body}"' if kind == "dialogue" else body, ""]
    L += ["--- SCRIPT END ---", ""]
    text = "\n".join(L)
    if crlf:
        text = text.replace("\n", "\r\n")
    return ("﻿" + text) if bom else text


def _t(x: float) -> str:
    m, s = divmod(round(x * 100), 6000)
    return f"{m:02d}:{s // 100:02d}.{s % 100:02d}"
