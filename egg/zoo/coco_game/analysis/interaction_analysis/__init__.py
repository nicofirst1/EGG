CTED = "correct_tartget==distr"
WTED = "wrong_tartget==distr"
CTND = "correct_tartget!=distr"
WTND = "wrong_tartget!=distr"

Tot = "total"
Crr = "correct"
Acc = "accuracy"
Frq = "frequency"

OCL = "other_classes_num"
TF = "target_freq"

PSC = "precision_sc"
POC = "precision_oc"
ARt = "ambiguity_rate"
ARc = "ambiguity_richness"
CR = "class_richness"

Sy = "symbols"
Se = "sequences"
SyCR = f"{Sy}_{CR}"
SeCR = f"{Se}_{CR}"

SeS = f"{Se}_specificity"
ISeU = f"intraclass_{Se}_uniqueness"

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
    CR: f"Given a target class *TRG* and a set of distractors associated with the target *DST_trg*, the {CR} is the number of {Sy}/{Se} used for *DST_trg* divided by the total amount of {Sy}/{Se} for all the target classes",
    ARc: f"The {ARc} is ratio between number of unique {Se} used when target=distractor  and when target!= distractor. ",
    SeS: f"The {SeS} is the proportion of {Se} used mainly for one supercategory in relation to the total length of all {Se}. ",
    ISeU: f"The {ISeU} is calculated onto the {SeS}. For all the {Se} in one superclass it counts the time a {Se} is used with more than one class. For example a superclass with {ISeU} zero has an unique {Se} for each class, while a  {ISeU} of 1 means that all the symbols are shared among each class.",
}
EXPLENATIONS[SeCR] = f"{SeCR} refers to the {CR} on the {Se}.\nThe {CR} is defined as: {EXPLENATIONS[CR]}."
