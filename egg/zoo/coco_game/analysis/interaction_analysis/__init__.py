CTED = "correct_tartget==distr"
WTED = "wrong_tartget==distr"
CTND = "correct_tartget!=distr"
WTND = "wrong_tartget!=distr"

Tot = "total"
Crr = "correct"
Acc = "accuracy"
Frq = "frequency"

OCL = "other_classes_len"
TF = "target_freq"

PSC = "precision_sc"
POC = "precision_oc"
ARt = "ambiguity_rate"
ARc = "ambiguity_richness"
CR = "class_richness"

Sy = "Symbol"
Se = "Sequence"
SyCR = f"{Sy}{CR}"
SeCR = f"{Se}{CR}"

EXPLENATIONS = {
    CTED: "Number of correct predictions when the target and the distractor belong to the same class",
    CTND: "Number of incorrect predictions when the target and the distractor belong to the same class",
    WTED: "Number of correct predictions when the target and the distractor belong to different classes",
    WTND: "Number of incorrect predictions when the target and the distractor belong to different classes",
    OCL: "Number of classes present in an image with the current target. Some classes such as person appear with many more objects in the image (i.e. other persons), while some like cat have usually only one object, the cat.",
    TF: "The frequency of the current target in the dataset.",
    PSC: f"Precision of the prediction when the disctractor and target belong to the same class.\nDerived from the formula :\n ```{CTED} / ({WTED}+{CTED})```",
    POC: f"Precision of the prediction when the disctractor and target belong to different classes.\nDerived from the formula :\n ```{CTND} / ({WTND}+{CTND})```",
    ARt: f"The {ARt} is the number of times the class happens both as a target and a distractors divided the total number of appearances.\nThe formula is derived as follos:\n```({CTED}+{WTED})/{Tot}```",
    CR: f"The {CR} is the number of {Sy}/{Se} used for specific class divided the total amount of {Sy}/{Se} for that class",
    ARc: f"The {ARc} is ratio between number of unique {Se} used when target=distractor  and when target!= distractor. ",
}
