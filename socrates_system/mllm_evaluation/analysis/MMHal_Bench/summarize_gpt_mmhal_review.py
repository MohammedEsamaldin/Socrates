import os
import sys
import glob
import numpy as np
import json
try:
    import jsonlines
except ImportError:
    jsonlines = None

def read_json(path):
    """Read and return the parsed contents of a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed Python object (typically a list or dict).
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data

def read_jsonl(jsonl_file):
    """Read and return all records from a JSONL file.

    Args:
        jsonl_file: Path to the JSONL file.

    Returns:
        A list of parsed record dicts.

    Raises:
        ImportError: If the ``jsonlines`` package is not installed.
    """
    data = []
    if jsonlines is None:
        raise ImportError("jsonlines package not installed; install it or avoid using read_jsonl.")
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data


def cal_informative(path, return_meta=False):
    """Compute the MMHal informativeness score from a GPT-4 review JSON file.

    Args:
        path: Path to the GPT-4 review JSON (list of response records).
        return_meta: If ``True``, also return the per-sample informativeness list.

    Returns:
        The mean informativeness percentage (0–100), or a tuple
        ``(informativeness_list, mean_informativeness)`` when ``return_meta=True``.
    """
    responses = read_json(path)

    def extract_score(item):
        """Extract a numeric rating from a single GPT-4 review record.

        Args:
            item: A response dict that may contain ``'rating'``, ``'judge_text'``,
                or an OpenAI ``choices[0].message.content`` field.

        Returns:
            An integer rating in [0, 6], defaulting to 0 if unparseable.
        """
        # Prefer explicit numeric rating if present
        if isinstance(item, dict) and 'rating' in item:
            try:
                return int(item['rating'])
            except Exception:
                pass
        # Try judge_text formatted ratings
        txt_fields = []
        if isinstance(item, dict):
            if 'judge_text' in item and isinstance(item['judge_text'], str):
                txt_fields.append(item['judge_text'])
            # Original OpenAI-style schema
            try:
                txt_fields.append(item['choices'][0]['message']['content'])
            except Exception:
                pass
        for txt in txt_fields:
            low = txt.lower()
            scores_found = []
            for s in range(7):
                if f'rating: {s}' in low:
                    scores_found.append(s)
            if len(scores_found) == 1:
                return scores_found[0]
        # Fallback
        return 0

    scores = [extract_score(r) for r in responses]

    informativeness = []
    for s in scores:
        if s >= 3:
            informativeness.append(s-3)
        else:
            informativeness.append(s)

    mean_informativeness = np.mean(informativeness)/3 * 100
    print("Informativeness: {:.2f}".format(mean_informativeness))

    if return_meta:
        return informativeness, mean_informativeness
    else:
        return mean_informativeness

def cal_mmhalscore(path):
    """Compute the MMHal hallucination score from a GPT-4 review JSON file.

    Args:
        path: Path to the GPT-4 review JSON (list of response records).

    Returns:
        A tuple ``(hallucination_rate, mean_score)`` where ``hallucination_rate``
        is the fraction of samples flagged as hallucinated and ``mean_score``
        is the mean rating across all samples.
    """
    responses = read_json(path)

    def extract_score(item):
        """Extract a numeric rating from a GPT-4 review record inside cal_mmhalscore.

        Args:
            item: A response record dict.

        Returns:
            An integer rating in [0, 6], defaulting to 0 if unparseable.
        """
        if isinstance(item, dict) and 'rating' in item:
            try:
                return int(item['rating'])
            except Exception:
                pass
        txt_fields = []
        if isinstance(item, dict):
            if 'judge_text' in item and isinstance(item['judge_text'], str):
                txt_fields.append(item['judge_text'])
            try:
                txt_fields.append(item['choices'][0]['message']['content'])
            except Exception:
                pass
        for txt in txt_fields:
            low = txt.lower()
            scores_found = []
            for s in range(7):
                if f'rating: {s}' in low:
                    scores_found.append(s)
            if len(scores_found) == 1:
                return scores_found[0]
        return 0

    def extract_hallu(item, score):
        """Determine whether a sample contains a hallucination.

        Args:
            item: A response record dict.
            score: The already-parsed numeric rating for this record.

        Returns:
            ``1`` if hallucination is detected, ``0`` otherwise.
        """
        if isinstance(item, dict) and 'hallucination' in item:
            try:
                return int(item['hallucination'])
            except Exception:
                pass
        # Derive from score if not provided: score < 3 => hallucination
        return 0 if score >= 3 else 1

    def extract_qtype(item, idx):
        """Extract or infer the question type label for grouping.

        Args:
            item: A response record dict.
            idx: Zero-based index of the record, used for cyclic fallback grouping.

        Returns:
            A question type string, or ``'type_<idx % 8>'`` as a fallback.
        """
        if isinstance(item, dict) and 'question_type' in item and item['question_type'] is not None:
            return str(item['question_type'])
        # Fallback to cyclical grouping of 8 types when possible
        return f"type_{idx % 8}"

    scores = [extract_score(r) for r in responses]

    hallucination = [extract_hallu(r, s) for r, s in zip(responses, scores)]

    # Group by question_type when available; otherwise fallback to 8 cyclic groups
    qtype_buckets = {}
    for i, item in enumerate(responses):
        qtype = extract_qtype(item, i)
        qtype_buckets.setdefault(qtype, []).append(scores[i])

    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))
    per_type_str = ','.join([f"{k}:{round(sum(v)/len(v), 2)}" for k, v in qtype_buckets.items()])
    print('Average score for each question type:', per_type_str, flush=True)

if __name__ == '__main__':
    base_path = sys.argv[1]
    print(base_path)

    review_files = []
    if os.path.isfile(base_path) and base_path.endswith('.json'):
        review_files = [base_path]
    else:
        # Treat as directory; process all json files recursively
        base_dir = base_path if base_path.endswith(os.sep) else base_path + os.sep
        patterns = ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*']
        f_list = sum([list(glob.glob(base_dir + p)) for p in patterns], [])
        review_files = [x for x in f_list if x.endswith('.json')]

    for file in review_files:
        print("===>", file)
        cal_informative(file)
        cal_mmhalscore(file)