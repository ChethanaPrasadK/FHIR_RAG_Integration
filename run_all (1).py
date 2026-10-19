"""Run Lab 5 end-to-end for one or more patients and build the submission files.

Usage (from the lab folder, after `pip install -r requirements_google_colab.txt`):
    python run_all.py              # all 9 patients (skips ones already in submission/)
    python run_all.py 2 3 5        # only these patients
    python run_all.py --force      # redo everything
Outputs: submission/experiment<N>/experiment_result.json, submission.json, submission.zip
Needs internet access to huggingface.co (Flan-T5 + nomic embeddings), e.g. Google Colab.
"""
import json, os, sys, shutil, logging, warnings
warnings.filterwarnings("ignore")

from experiment_helpers import load_experiment_config
import experiments as E

REQUIRED = [1, 4, 8]
ALL = list(range(1, 10))
TOLERANCE = 0.05


def evaluate(student_data, openai_answers):
    import nltk
    nltk.download('wordnet', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge import Rouge
    import bert_score
    smooth, rouge = SmoothingFunction().method4, Rouge()
    results = {}
    for key, qa in student_data['question_answering'].items():
        ref = openai_answers['question_answering'].get(key, {}).get('openai_answer', "")
        if not ref:
            continue
        hyp = qa['model_answer']
        h, r = hyp.split(), ref.split()
        try: bleu = sentence_bleu([r], h, smoothing_function=smooth)
        except Exception: bleu = 0.0
        try: rl = rouge.get_scores(' '.join(h), ' '.join(r), avg=True)['rouge-l']['f']
        except Exception: rl = 0.0
        try: met = meteor_score([r], h)
        except Exception: met = 0.0
        try: bf = bert_score.score([hyp], [ref], lang="en", verbose=False)[2].mean().item()
        except Exception: bf = 0.0
        if bf > 0:
            rel = bf if (bleu > 0.1 or rl > 0.1 or met > 0.1) else bf * 0.3
        else:
            rel = 0.0
        results[key] = {"question": qa.get('question'), "model_answer": hyp, "openai_answer": ref,
                        "bleu_score": bleu, "rouge_l_f_score": rl, "meteor_score": met,
                        "bert_score": bf, "final_percentage": rel * 100}
    return results


def run_patient(n):
    config = load_experiment_config('experiments.json')
    t1 = config['tasks']['task1_read_fhir_bundle']
    t1.update({"bundle_name": f"fhir_bundles/patient{n}.json", "transaction_type": "transaction",
               "resource_type": "DocumentReference", "resource_attribute": "attachment",
               "additional_resource_type": "DiagnosticReport",
               "additional_resource_attribute": "presentedForm"})
    t2 = config['tasks']['task2_vector_db_and_genai_model_config']
    t2.update({"bundle_name": t1["bundle_name"], "temperature": 0.3,
               "embedding_name": "nomic-ai/nomic-embed-text-v1",
               "input_clinical_note_file": "clinical_notes_output.txt",
               "model_name": "google/flan-t5-base"})

    E.main_read_fhir_bundle(config)
    E.main_vector_db_and_genai_model_config(config)
    E.main_question_answering(config, "No")

    student = json.load(open(f'patient{n}_result.json'))
    ref = json.load(open(f'openai_answer/{n}.json'))
    results = evaluate(student, ref)
    out = f'submission/experiment{n}'
    os.makedirs(out, exist_ok=True)
    json.dump(results, open(f'{out}/experiment_result.json', 'w'), indent=4)
    print(f"patient {n}: wrote {out}/experiment_result.json")
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify():
    ok = True
    for n in REQUIRED:
        try:
            sub = json.load(open(f'submission/experiment{n}/experiment_result.json'))
            res = json.load(open(f'results/experiment{n}/experiment_result.json'))
        except FileNotFoundError:
            print(f"experiment {n}: missing file"); ok = False; continue
        for q, d in sub.items():
            for m in ["bleu_score", "rouge_l_f_score", "meteor_score", "bert_score"]:
                s, r = d.get(m, 0), res.get(q, {}).get(m, 0)
                if s and r and s < r - TOLERANCE:
                    print(f"experiment {n} {q} {m}: {s:.3f} < {r - TOLERANCE:.3f}"); ok = False
        # no empty values (openai_answer exempt)
        for q, d in sub.items():
            for k, v in d.items():
                if k != "openai_answer" and (v is None or v == ""):
                    print(f"experiment {n} {q}.{k} is empty"); ok = False
    missing = [n for n in ALL if not os.path.exists(f'submission/experiment{n}/experiment_result.json')]
    sj = json.load(open('submission.json'))
    sj['state'] = '1' if ok else sj.get('state', '0')
    json.dump(sj, open('submission.json', 'w'), indent=4)
    if missing:
        print(f"Missing result files for patients {missing} -> run: python run_all.py {' '.join(map(str, missing))}")
    if not ok:
        print("Some required (1, 4, 8) checks failed - see messages above")
    if not missing:
        if os.path.exists('submission.zip'): os.remove('submission.zip')
        shutil.make_archive('submission', 'zip', 'submission')
        print("submission.zip created with experiments 1-9" + ("" if ok else " (WARNING: required checks failed)"))


if __name__ == "__main__":
    args = sys.argv[1:]
    force = '--force' in args
    patients = [int(a) for a in args if a.isdigit()] or ALL
    for p in patients:
        done = os.path.exists(f'submission/experiment{p}/experiment_result.json')
        if done and not force and not [a for a in args if a.isdigit()]:
            print(f"patient {p}: already done, skipping (use --force to redo)")
            continue
        run_patient(p)
    verify()
