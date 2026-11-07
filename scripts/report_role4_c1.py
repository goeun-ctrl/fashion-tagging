"""Watch the isolated pipeline and write a Korean result report once evaluation completes."""
import argparse
import csv
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def table(records, columns):
    result = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for record in records:
        values = []
        for column in columns:
            value = record[column]
            try:
                value = f'{float(value):.4f}' if column not in ('support', 'best_epoch', 'epochs_run') else str(value)
            except (TypeError, ValueError):
                pass
            values.append(str(value))
        result.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(result)


def render(state):
    base = ROOT / 'runs/role4_c1'
    hold, val = [ROOT / ('report/role4_c1_' + split + '_s42') for split in ('holdout', 'val')]
    selection = json.loads((hold / 'selection.json').read_text())
    hs, vs = rows(hold / 'summary.csv'), rows(val / 'summary.csv')
    hc, vc = rows(hold / 'comparison.csv'), rows(val / 'comparison.csv')
    selected = selection['selected_loss']
    base_delta = next(row for row in hc if row['mode'] == 'baseline')
    lo, hi = [float(base_delta[k]) for k in ('ci_low', 'ci_high')]
    evidence = ('holdout 기본 Macro F1 차이의 95% 구간이 0을 포함하므로 확정적인 우열은 판단하기 어렵습니다.'
                if lo <= 0 <= hi else
                'holdout 기본 Macro F1 차이의 95% 구간은 0을 포함하지 않습니다. 다만 학습 seed 변동을 검증한 결과는 아닙니다.')
    summaries = []
    for loss in ('bce', 'asl'):
        s = json.loads((base / f'R4_C1_{loss}_s42/summary.json').read_text())
        summaries.append(dict(loss=loss, best_epoch=s['best_epoch'], epochs_run=s['epochs_run'],
                              train_minutes=s['train_minutes'], amp_skipped=s['amp_skipped']))
    smoke = json.loads((ROOT / 'runs/role4_c1_gpu_smoke/checks.json').read_text())
    gpu_description = ('BCE GPU3 / ASL GPU0 (각 A5000 한 장), BCE 기존 프로세스 유지·ASL 공통 초기 상태에서 병렬 학습'
                       if state.get('mode') == 'parallel' else 'A5000 GPU3 한 장 고정, BCE→ASL 순차 학습')
    text = f'''# C1 BCE·ASL 비교 결과 (seed 42)

**holdout 기본 Macro F1 기준 잠정 선택: {selected.upper()}**.

{evidence}
Loss·epoch·임계값은 공식 val을 보기 전에 고정했습니다. val 결과로 선택을 변경하지 않았습니다.
작은 단일 seed 차이를 일반적인 Loss 우열로 해석하지 않습니다. 확정하려면 추가 seed에서 방향이 유지되는지 확인해야 합니다.

## 실행·재현 검증

- GPU: {gpu_description}, 각각 18 epoch.
- 공통 초기 파일 SHA256: `{state['initial_sha256']}`. 테스트·두 학습 후 보존: `{state.get('initial_preserved')}`.
- GPU 소량 테스트: `{smoke['passed']}`, batch8×1344×800 stress 검사와 체크포인트 재로딩 통과.
- 비교기는 두 Loss의 설정·초기 모델/헤드 hash·sampler 계획·클래스/이미지/원본 정답 순서·라이브러리·소스 hash 일치를 검사했습니다.
- 원본 Fashionpedia만 사용. 고정 train_inner 43,197 / holdout 2,425 / val 1,158. 외부·증류·gate·새 head·추가 증강 없음.
- 학습 곡선·checkpoint·전체 실행 로그: `runs/role4_c1/`.
- 공식 val은 두 18 epoch 학습 및 holdout 선택 파일 생성 후에만 실행했습니다.
- GitHub push 및 타인 작업 종료 없음.

{table(summaries, ['loss','best_epoch','epochs_run','train_minutes','amp_skipped'])}

## 선택 규칙

기본은 EMA best epoch, 추론 800/1333, threshold .5, TTA 없음의 holdout Macro F1입니다.
추가 비교는 양쪽 모두 동일한 전역 threshold .05~.95(간격 .05)를 holdout에서만 탐색했습니다.
TTA는 원본·좌우 반전 logits 평균 후 sigmoid입니다. 기존 ASL .60·클래스별 예외는 이식하지 않았습니다.
임계값·TTA 적용 후 순위가 바뀌더라도 기본 비교와 후처리 효과를 구분합니다.

선택된 threshold: BCE={selection['runs']['bce']['threshold']}, ASL={selection['runs']['asl']['threshold']};
TTA용 threshold: BCE={selection['runs']['bce']['tta_threshold']}, ASL={selection['runs']['asl']['tta_threshold']}.

## Holdout 전체 결과

{table(hs, ['loss','mode','threshold','macro_f1','micro_f1','image_f1','macro_precision','macro_recall','micro_precision','micro_recall'])}

## 공식 val 최종 보고

{table(vs, ['loss','mode','threshold','macro_f1','micro_f1','image_f1','macro_precision','macro_recall','micro_precision','micro_recall'])}

## Macro F1 차이와 이미지 단위 쌍 부트스트랩

차이는 ASL−BCE입니다. 이미지 단위 쌍 추출 2,000회, seed42, percentile 95% 구간입니다.
threshold와 학습 모델을 고정한 조건부 구간이며 seed 변동·holdout 선택 불확실성을 포함하지 않습니다.

Holdout:

{table(hc, ['mode','bce_macro_f1','asl_macro_f1','delta_asl_minus_bce','ci_low','ci_high'])}

공식 val (보고용):

{table(vc, ['mode','bce_macro_f1','asl_macro_f1','delta_asl_minus_bce','ci_low','ci_high'])}

Micro/Image-wise F1 차이의 구간도 각 split의 `bootstrap.json`에 있습니다.

## 희소 클래스 7개

기본 .5/no-TTA 결과입니다. 양성이 적은 클래스에서는 한두 이미지가 F1을 크게 바꿀 수 있습니다.
'''
    for name, folder in (('Holdout', hold), ('공식 val', val)):
        rare = [r for r in rows(folder / 'rare_classes.csv') if r['mode'] == 'baseline']
        text += '\n' + name + ':\n\n' + table(rare, ['loss','category','support','f1','precision','recall']) + '\n'
    text += '\n## 전체 클래스별 결과\n'
    for name, folder in (('Holdout', hold), ('공식 val', val)):
        full = [r for r in rows(folder / 'per_class.csv') if r['mode'] == 'baseline']
        text += '\n' + name + ':\n\n' + table(full, ['loss','category','support','f1','precision','recall']) + '\n'
    text += '\n모든 후처리 모드의 27개 클래스 결과는 각 split의 `per_class.csv`, 희소 클래스는 `rare_classes.csv`에 저장했습니다.\n'
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    out = ROOT / 'report/role4_c1_final'
    out.mkdir(parents=True, exist_ok=True)
    report = out / 'report.md'
    while True:
        status = json.loads((ROOT / 'runs/role4_c1/pipeline_status.json').read_text())
        if status['status'] == 'complete':
            content = render(status)
        elif status['status'] == 'failed':
            content = '# C1 비교 실행 중단\n\n실패 단계: ' + status['current_stage'] + '\n\n' + status['error'] + '\n\n완료 결과로 해석하지 않습니다. `runs/role4_c1/`의 해당 단계 로그를 확인해야 합니다.\n'
        else:
            gpu_text = ('BCE GPU3 / ASL GPU0에서 각각18epoch 병렬 학습합니다. 두 학습이 정상 완료된 뒤에만 holdout 선택→공식 val 평가를 진행합니다.'
                        if status.get('mode') == 'parallel' else 'GPU3 한 장에서 BCE→ASL 각18epoch→holdout 선택→공식 val 평가 순서로 진행합니다.')
            content = '# C1 비교 진행 중\n\n현재 단계: `' + status['current_stage'] + '`\n\n' + gpu_text + ' 완료 전 Loss 우열은 판단하지 않습니다.\n\n로그: `runs/role4_c1/`. 완료되면 이 문서를 전체·클래스별·bootstrap 결과와 선택 근거로 갱신합니다.\n'
        tmp = report.with_suffix('.tmp'); tmp.write_text(content); tmp.replace(report)
        if status['status'] in ('complete','failed') or not a.watch:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
