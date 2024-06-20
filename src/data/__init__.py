from .tab import Dataset as TABDataset, \
    Evaluator as TABEvaluator, \
    NMS as TABNMS, \
    set_default_config_ as config_tab, \
    format as format_tab, \
    preprocess as preprocess_tab


DATASET = {
    'TAB': TABDataset,
}

EVALUATOR = {
    'TAB': TABEvaluator,
}

NMS = {
    'TAB': TABNMS,
}

CONFIG = {
    'TAB': config_tab,
}

FORMAT = {
    'TAB': format_tab,
}

PREPROCESS = {
    'TAB': preprocess_tab,
}
