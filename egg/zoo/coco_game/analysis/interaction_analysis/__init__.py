CTED = "V_trg==dstr"
WTED = "X_trg==dstr"
CTND = "V_trg!=dstr"
WTND = "X_trg!=dstr"

Tot = "total"
Crr = "correct"
Acc = "accuracy"
Frq = "freq"

OCL = "othr_clss_num"
TF = "trg_freq"

PSC = "prec_sc"
POC = "prec_oc"
ARt = "ambgty_rate"
ARc = "ambgty_rchns"
CR = "class_rchns"

Sy = "sym"
Se = "seq"
SyCR = f"{Sy}_{CR}"
SeCR = f"{Se}_{CR}"

SeS = f"{Se}_spec"
ISeU = f"intracls_{Se}_uniq"

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
    CR: f"The class richness is the number of {Se}/{Sy} associated with a target class normalized by all the {Se}/{Sy} for every target class. ",
    ARc: f"The {ARc} is ratio between number of unique {Se} used when target=distractor  and when target!= distractor. ",
    SeS: f"The {SeS} is the proportion of {Se} used mainly for one superclass in relation to the total length of all {Se}. ",
    ISeU: f"The {ISeU} is calculated onto the {SeS}. For all the {Se} in one superclass it counts the time a {Se} is used with more than one class. For example a superclass with {ISeU} zero has an unique {Se} for each class, while a  {ISeU} of 1 means that all the symbols are shared among each class.",
}
EXPLENATIONS[SeCR] = f"{SeCR} refers to the {CR} on the {Se}.\nThe {CR} is defined as: {EXPLENATIONS[CR]}."
