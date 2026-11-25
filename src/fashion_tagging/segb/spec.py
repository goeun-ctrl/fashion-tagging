"""Fixed definitions for redesign B (docs/plan/재설계B_구현계획.md)."""

# group-wise softmax: each group = "none" + its classes (category ids of categories.txt, 0..26)
GROUPS = [
    [0, 1, 2, 3, 4, 5, 9, 12],                  # upper / outer: shirt, top, sweater, cardigan, jacket, vest, coat, cape
    [6, 7, 8, 10, 11],                          # lower / one-piece: pants, shorts, skirt, dress, jumpsuit
    [20, 21, 22, 23],                           # legs / feet: leg warmer, tights, sock, shoe
    [13, 14, 15, 16, 17, 18, 19, 24, 25, 26],   # accessories
]
# class id -> (group index, 1-based position inside the group; 0 = none)
CLASS_TO_GROUP = {c: (g, j + 1) for g, cs in enumerate(GROUPS) for j, c in enumerate(cs)}
GROUP_SIZES = [len(g) + 1 for g in GROUPS]      # channels per group including "none"
N_SEG = sum(GROUP_SIZES)                        # 31 output channels
# Fashionpedia part ids used as an auxiliary sigmoid output: sleeve, neckline, pocket, collar, zipper, lapel, hood
PARTS = [31, 33, 32, 28, 35, 29, 27]
STRIDE = 4                                      # output stride of the segmentation map (one cell = 4 input pixels)
IGNORE = 255
assert sorted(CLASS_TO_GROUP) == list(range(27))
