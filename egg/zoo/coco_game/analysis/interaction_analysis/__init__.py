_target = "trg"
_distractor = "dstr"

NObj = "number_objects"

CTED = f"V_{_target}=={_distractor}"
WTED = f"X_{_target}=={_distractor}"
CTND = f"V_{_target}!={_distractor}"
WTND = f"X_{_target}!={_distractor}"

Tot = "total"
Crr = "correct"
Acc = "accuracy"
Frq = "freq"

OCL = "othr_clss_num"
TF = f"{_target}_{Frq}"

PSC = "prec_sc"
POC = "prec_oc"
_ambiguity = "ambg"

ARt = f"{_ambiguity}_rate"
ARc = f"{_ambiguity}_rchns"
ARcP = f"{ARc}_perc"
CR = "cls_rchns"

Sy = "sym"
Se = "seq"
SyCR = f"{Sy}_{CR}"
SeCR = f"{Se}_{CR}"
SeNm = f"{Se}_num"

ClsCmt = "cls_comity"

SSeS = f"supercls_{Se}_spec"
CSeS = f"cls_{Se}_spec"
SScS = f"shared_supercls_{Se}"
ISeS = f"intracls_{Se}_spcf"

DEFINITIONS = {
    CTED: "Number of correct predictions when the target and the distractor belong to the same class",
    CTND: "Number of incorrect predictions when the target and the distractor belong to the same class",
    WTED: "Number of correct predictions when the target and the distractor belong to different classes",
    WTND: "Number of incorrect predictions when the target and the distractor belong to different classes",
    OCL: "Number of classes present in an image with the current target. Some classes such as person appear with many"
         " more objects in the image (i.e. other persons), while some like cat have usually only one object, the cat.",
    TF: "The frequency of the current target in the dataset.",
    PSC: f"Precision of the prediction when the disctractor and target belong to the same class.\n"
         f"Derived from the formula :\n ```{CTED} / ({WTED}+{CTED})```",
    POC: f"Precision of the prediction when the disctractor and target belong to different classes.\n"
         f"Derived from the formula :\n ```{CTND} / ({WTND}+{CTND})```",
    ARt: f"The ambiguity rate is the number of times the class happens both as a target and a distractors divided the"
         f" total number of appearances.\nThe formula is derived as follos:\n```({CTED}+{WTED})/{Tot}```",
    CR: f"The class richness is the number of sequences/symbols associated with a target"
        f" class normalized by the number of targets. ",
    ARc: f"The ambiguity richness estimates the number of sequences used for a target when target==distractor"
         f" divided by the number of sequences used when target!=distractor (for the same target).\n"
         f"For example, given the class cat which appears with dog, cat and bike in the dataset, the ambiguity richness is equal to:\n"
         f"```len(Seq(cat,cat))/[len(Seq(cat,cat)+Seq(cat,dog) +Seq(cat, bike))]```\n"
         f"Where `Seq(i,j)` returns the sequences when the target i appears together with the distractor j.",
    ARcP: f"The *ambiguity richness perc* follows the same pattern as the standar ambiguity richness but instead of "
          f"counting how many sequences are used with the `Seq(i,j)` function it counts the number of times the each sequence is used:\n"
          f"```SeqPerc(i,j)=Sum(Seq(i,j))```",
    SSeS: f"The *super class sequence specificity* is the proportion of sequences used mainly (more than 99% of the times)"
          f" for one superclass divided by the number of sequences.\nExample:\n"
          f"Having a sequence specificity of 10% means that 10% of all the sequences appear only with one class.\n",

    CSeS: f"The *class sequence specificity* is the number of sequences used in one class only divided by the total number of sequences.\n",
    SScS: f"The *shared superclass sequences* is calculated onto the sequence specificity. "
          f"It gathers information regarding both the superclass and the class. "
          f"For all the sequences in one superclass it counts the time a sequence is used with more than one class (shared).\n"
          f"For example a superclass with *shared superclass sequences* zero has an unique sequence for each class, while an "
          f"*class sequence specificity* of 1 means that all the sequences are shared among all the classes of a superclass.",
    ISeS: f"The *intra class specificity* is a variant of the *class sequence specificity*.\n"
          f"It is defined as the number of unique class sequences divided by the class target frequency. ",
    Frq: "Number of times an object appears as target, distractor or in the other classes, divided by all the objects in the dataset",
    ClsCmt: "Given a target class t, the class comity is the number of other classes d which appears together with t"
            " divided by the total number of classes.\n"
            "It is a measure of how likely a class is to appear with another one or alone."

}

PRINT_DEF = ["\n# Metrics definitions\n"] + [f"- **{k}** : {v}\n\n" for k, v in DEFINITIONS.items()]
