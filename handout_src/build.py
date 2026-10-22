# ──────────────────────────────────────────────
# 타이타닉 데이터 분석 교안(PDF) 빌드 스크립트
#   1) titanic_kaggle.csv 로 분석을 다시 돌려 그림(figures/*.png)을 만들고
#   2) handout.html 을 크롬/엣지 헤드리스로 인쇄해 PDF 를 만듭니다.
# 실행 : python handout_src/build.py   (저장소 루트에서)
# ──────────────────────────────────────────────
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
PDF_OUT = ROOT / '타이타닉_데이터분석_교안.pdf'

# 색상 : 생존 = 파랑, 사망 = 주황 (모든 그림에서 같은 의미로 고정)
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2 = '#0b0b0b', '#52514e'

sns.set_theme(style='whitegrid', font='Malgun Gothic')
plt.rcParams.update({
    'axes.unicode_minus': False,
    'axes.edgecolor': '#d4d3cf', 'grid.color': '#ebeae6',
    'axes.titleweight': 'bold', 'axes.titlesize': 13,
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.grid.axis': 'y',
})


# ──────────────────────────────────────────────
# 데이터 준비 (교안 3·6 장과 같은 방법)
# ──────────────────────────────────────────────
raw = pd.read_csv(ROOT / 'titanic_kaggle.csv')
df = raw.copy()
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df['AgeMissing'] = df['Age'].isnull().astype(int)
df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
df = df.drop(columns=['Cabin'])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
df['Title'] = np.where(df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master']), df['Title'], 'Rare')
df['성별'] = df['Sex'].map({'female': '여성', 'male': '남성'})
df['등급'] = df['Pclass'].astype(str) + '등석'
overall = df['Survived'].mean()


def save(fig, name):
    fig.savefig(FIG / f'{name}.png')
    plt.close(fig)


def pct_labels(ax, fmt='{:.0%}'):
    """막대 끝에 값 표시 (값은 글자색으로, 막대색 아님)"""
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v) if not np.isnan(v) else '' for v in c.datavalues],
                     padding=3, fontsize=9, color=INK2)


def rate_axis(ax, top=1.08):
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    sns.despine(ax=ax, left=True)


# 그림 3-1 결측치 비율
miss = (raw.isnull().mean()).sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.barh(miss.index[::-1], miss.values[::-1], color=BLUE, height=0.5)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
for i, v in enumerate(miss.values[::-1]):
    ax.text(v + 0.01, i, f'{v:.1%}  ({int(v * len(raw))}명)', va='center', fontsize=9, color=INK2)
ax.set_xlim(0, 1)
ax.set_title('열별 결측치 비율')
sns.despine(ax=ax, bottom=True)
save(fig, 'fig3_missing')

# 그림 7-1 생존자 수 · 성별 · 등급×성별
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
cnt = df['Survived'].value_counts().sort_index()
axes[0].bar(['사망 (0)', '생존 (1)'], cnt.values, color=[ORANGE, BLUE], width=0.55)
axes[0].bar_label(axes[0].containers[0], padding=3, fontsize=9, color=INK2)
axes[0].set_title('생존자 수'); sns.despine(ax=axes[0], left=True)
s = df.groupby('성별')['Survived'].mean().reindex(['여성', '남성'])
axes[1].bar(s.index, s.values, color=BLUE, width=0.5)
pct_labels(axes[1]); rate_axis(axes[1]); axes[1].set_title('성별 생존율')
sns.barplot(data=df, x='등급', y='Survived', hue='성별', hue_order=['여성', '남성'],
            order=['1등석', '2등석', '3등석'], errorbar=None, palette=[BLUE, AQUA], ax=axes[2])
pct_labels(axes[2]); rate_axis(axes[2]); axes[2].set_title('객실 등급 × 성별 생존율')
axes[2].set_xlabel(''); axes[2].set_ylabel(''); axes[2].legend(title='', frameon=False)
plt.tight_layout()
save(fig, 'fig7_overview')

# 그림 7-2 나이 분포 (원본 기준)
fig, ax = plt.subplots(figsize=(8, 3.6))
tmp = raw.dropna(subset=['Age']).assign(생존=lambda d: d['Survived'].map({0: '사망', 1: '생존'}))
sns.histplot(data=tmp, x='Age', hue='생존', hue_order=['사망', '생존'], multiple='stack', bins=30,
             palette=[ORANGE, BLUE], edgecolor='white', linewidth=1, ax=ax)
ax.set_title('생존 여부별 나이 분포 (결측치 채우기 전)'); ax.set_xlabel('나이'); ax.set_ylabel('승객 수')
sns.despine(ax=ax, left=True)
save(fig, 'fig7_age')

# 그림 7-3 등급별 요금 상자 그림
fig, ax = plt.subplots(figsize=(8, 3.6))
sns.boxplot(data=df, x='등급', y='Fare', order=['1등석', '2등석', '3등석'], color='#cfe0f5',
            linecolor=BLUE, flierprops=dict(marker='o', markersize=4, markerfacecolor=BLUE, alpha=.5), ax=ax)
ax.set_title('객실 등급별 요금 분포'); ax.set_xlabel(''); ax.set_ylabel('요금')
sns.despine(ax=ax, left=True)
save(fig, 'fig7_fare')

# 그림 7-4 가족 수별 생존율
fam = df.groupby('FamilySize')['Survived'].agg(['mean', 'count'])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(fam.index, fam['mean'], color=BLUE, lw=2, marker='o', ms=8, mec='white', mew=2)
ax.axhline(overall, ls='--', color=GRAY, lw=1)
ax.text(11, overall + 0.03, f'전체 평균 {overall:.1%}', ha='right', fontsize=9, color=INK2)
for x, (m, n) in fam.iterrows():
    ax.text(x, m + 0.04, f'{m:.0%} · {int(n)}명', ha='center', fontsize=8, color=INK2,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
ax.set_xticks(fam.index); rate_axis(ax, 1.0)
ax.set_title('가족 수별 생존율'); ax.set_xlabel('가족 수 (본인 포함)')
save(fig, 'fig7_family')

# 그림 7-5 상관관계 히트맵
corr_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'FamilySize']].copy()
corr_df.insert(1, 'Sex(여=1)', df['Sex'].map({'male': 0, 'female': 1}))
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, center=0,
            linewidths=2, linecolor='white', annot_kws={'size': 8}, ax=ax, cbar_kws={'shrink': .7})
ax.set_title('변수 간 상관관계')
save(fig, 'fig7_corr')

# 그림 8-1 규칙별 정확도
y = df['Survived'].values
is_female = df['Sex'] == 'female'
is_poor_bigfam = (df['Pclass'] == 3) & (df['FamilySize'] >= 5)
is_rich_boy = (df['Title'] == 'Master') & (df['Pclass'] <= 2)
acc = {
    '규칙 0\n모두 사망': (np.zeros(len(df), dtype=int) == y).mean(),
    '규칙 1\n여성 = 생존': (np.where(is_female, 1, 0) == y).mean(),
    '규칙 2\n규칙1 + 예외': (np.where((is_female & ~is_poor_bigfam) | is_rich_boy, 1, 0) == y).mean(),
}
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(acc.keys(), acc.values(), color=[GRAY, BLUE, BLUE], width=0.5)
pct_labels(ax, '{:.1%}'); rate_axis(ax, 1.0); ax.set_title('생존 예측 규칙별 정확도')
save(fig, 'fig8_rules')


# ──────────────────────────────────────────────
# 10 장 인사이트용 그림
# ──────────────────────────────────────────────
# 인사이트 1 : 성별 > 등급
pv = df.pivot_table(index='등급', columns='성별', values='Survived', aggfunc='mean')[['여성', '남성']]
fig, ax = plt.subplots(figsize=(8, 3.6))
pv.plot.bar(ax=ax, color=[BLUE, AQUA], width=0.7, edgecolor='white', linewidth=2, rot=0)
pct_labels(ax); rate_axis(ax)
ax.axhline(overall, ls='--', color=GRAY, lw=1)
ax.set_title('3등석 여성(50%)이 1등석 남성(37%)보다 많이 살았다'); ax.set_xlabel('')
ax.legend(title='', frameon=False, loc='upper right')
save(fig, 'ins1_sex_class')

# 인사이트 2 : 어린이 먼저? — 남자아이(Master)·여성의 등급별 생존율
grp = pd.DataFrame({
    '남자아이 (Master)': df[df['Title'] == 'Master'].groupby('등급')['Survived'].mean(),
    '여성': df[df['Sex'] == 'female'].groupby('등급')['Survived'].mean(),
    '성인 남성 (Mr)': df[df['Title'] == 'Mr'].groupby('등급')['Survived'].mean(),
})
fig, ax = plt.subplots(figsize=(8, 3.6))
grp.plot.bar(ax=ax, color=[BLUE, AQUA, ORANGE], width=0.75, edgecolor='white', linewidth=2, rot=0)
pct_labels(ax); rate_axis(ax, 1.12)
ax.set_title('"여성과 어린이 먼저" — 3등석에서는 절반도 지켜지지 않았다', pad=28); ax.set_xlabel('')
ax.legend(title='', frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
save(fig, 'ins2_children')

# 인사이트 3 : 가족 규모 × 등급
df['가족규모'] = pd.cut(df['FamilySize'], [0, 1, 4, 20], labels=['혼자', '2~4명', '5명 이상'])
fb = df.pivot_table(index='가족규모', columns='등급', values='Survived', aggfunc='mean', observed=True)
fc = df.pivot_table(index='가족규모', columns='등급', values='Survived', aggfunc='count', observed=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [1, 1.4]})
fall = df.groupby('가족규모', observed=True)['Survived'].mean()
axes[0].bar(fall.index.astype(str), fall.values, color=BLUE, width=0.5)
pct_labels(axes[0]); rate_axis(axes[0], 1.0); axes[0].set_title('가족 규모별 생존율')
cnt5 = fc.loc['5명 이상']
axes[1].bar(cnt5.index, cnt5.values, color=BLUE, width=0.5)
axes[1].bar_label(axes[1].containers[0],
                  labels=[f'{n}명 · 생존 {r:.0%}' for n, r in zip(cnt5.values, fb.loc['5명 이상'].values)],
                  padding=3, fontsize=9, color=INK2)
axes[1].set_title('5명 이상 대가족은 어느 등급이었나?'); axes[1].set_ylim(0, 65)
sns.despine(ax=axes[1], left=True)
plt.tight_layout()
save(fig, 'ins3_family')

# 인사이트 4 : 항구 — 전체 vs 등급 고정
emb_all = df.groupby('Embarked')['Survived'].mean()
emb_cls = df.pivot_table(index='Embarked', columns='등급', values='Survived', aggfunc='mean')
first_share = pd.crosstab(df['Embarked'], df['Pclass'], normalize='index')[1]
fem_share = pd.crosstab(df['Embarked'], df['Sex'], normalize='index')['female']
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
comp = pd.DataFrame({'1등석 비율': first_share, '여성 비율': fem_share})
comp.index = ['C 셰르부르', 'Q 퀸스타운', 'S 사우샘프턴']
comp.plot.bar(ax=axes[0], color=[BLUE, AQUA], width=0.7, edgecolor='white', linewidth=2, rot=0)
pct_labels(axes[0]); rate_axis(axes[0], 0.75); axes[0].set_title('항구별 승객 구성'); axes[0].legend(frameon=False)
e = emb_cls.copy(); e.index = comp.index
e.plot.bar(ax=axes[1], color=[BLUE, AQUA, ORANGE], width=0.75, edgecolor='white', linewidth=2, rot=0)
pct_labels(axes[1]); rate_axis(axes[1], 0.85); axes[1].set_title('등급을 고정하고 본 항구별 생존율', pad=28)
axes[1].legend(title='', frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
for ax in axes: ax.set_xlabel('')
plt.tight_layout()
save(fig, 'ins4_port')

# 인사이트 5 : 결측은 정보다
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
hc = df.groupby('등급')['HasCabin'].mean()
axes[0].bar(hc.index, hc.values, color=BLUE, width=0.5)
pct_labels(axes[0]); rate_axis(axes[0], 1.0); axes[0].set_title('객실 번호가 기록된 승객 비율')
am = df.pivot_table(index='등급', columns='AgeMissing', values='Survived', aggfunc='mean')
am.columns = ['나이 기록 있음', '나이 기록 없음']
am.plot.bar(ax=axes[1], color=[BLUE, GRAY], width=0.7, edgecolor='white', linewidth=2, rot=0)
pct_labels(axes[1]); rate_axis(axes[1], 0.85); axes[1].set_title('나이 기록 유무별 생존율')
axes[1].legend(frameon=False); axes[1].set_xlabel('')
plt.tight_layout()
save(fig, 'ins5_missing')

# 11 장 : 훈련 정확도 vs 교차검증 정확도 (과적합)
X = pd.DataFrame({'female': is_female.astype(int), 'Pclass': df['Pclass'], 'FamilySize': df['FamilySize'],
                  'Age': df['Age'], 'Fare': df['Fare']})
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for d in [2, 3, 4, 6, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=0)
    rows.append({'깊이': '제한 없음' if d is None else str(d),
                 '훈련 정확도': m.fit(X, y).score(X, y),
                 '검증 정확도(5-fold)': cross_val_score(m, X, y, cv=cv).mean()})
ov = pd.DataFrame(rows).set_index('깊이')
print(ov.round(3))
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(ov.index, ov['훈련 정확도'], color=ORANGE, lw=2, marker='o', ms=8, mec='white', mew=2, label='훈련 정확도 (본 데이터로 채점)')
ax.plot(ov.index, ov['검증 정확도(5-fold)'], color=BLUE, lw=2, marker='o', ms=8, mec='white', mew=2, label='검증 정확도 (안 본 데이터로 채점)')
for i, (a, b) in enumerate(ov.values):
    ax.text(i, a + 0.012, f'{a:.1%}', ha='center', fontsize=8, color=INK2)
    ax.text(i, b - 0.03, f'{b:.1%}', ha='center', fontsize=8, color=INK2)
ax.set_ylim(0.7, 1.02); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel('결정 트리 깊이'); ax.set_title('복잡한 모델일수록 "본 문제"만 잘 푼다')
ax.legend(frameon=False, loc='upper left'); sns.despine(ax=ax, left=True)
save(fig, 'adv_overfit')

print('그림 생성 완료 :', len(list(FIG.glob('*.png'))), '개')


# ──────────────────────────────────────────────
# HTML → PDF (크롬 또는 엣지 헤드리스 인쇄)
# ──────────────────────────────────────────────
candidates = [
    shutil.which('chrome'), shutil.which('msedge'),
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    shutil.which('google-chrome'), shutil.which('chromium'),
]
browser = next((c for c in candidates if c and os.path.exists(c)), None)
if browser is None:
    raise SystemExit('크롬/엣지를 찾지 못했습니다. handout.html 을 브라우저에서 열어 PDF 로 인쇄하세요.')
subprocess.run([browser, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                f'--print-to-pdf={PDF_OUT}', (HERE / 'handout.html').as_uri()], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('PDF 생성 :', PDF_OUT)
