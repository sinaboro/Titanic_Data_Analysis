# %% [markdown]
# # 🚢 타이타닉 데이터 분석 — 예제 코드 (강사·학생 공통)
#
# > 이 파일은 **교안(Word/PDF)의 각 장 번호와 똑같은 순서**로 구성되어 있습니다. 교안을 보면서 셀을 하나씩 실행해 보세요.
#
#
# > **선수 학습** : 넘파이(NumPy), 판다스(Pandas) 기초
# > **데이터** : 깃허브의 `titanic_kaggle.csv` (타이타닉호 승객 891명의 정보와 생존 여부) — 코드에서 바로 내려받습니다.
#
# ## 오늘의 질문
# **"타이타닉호에서 누가 살아남았을까? 그리고 그 이유는 무엇일까?"**
#
# 지금까지 배운 넘파이/판다스 문법을 **실제 데이터**에 적용해 보면서,
# 데이터 분석의 전체 흐름을 처음부터 끝까지 경험해 봅니다.
#
# ## 데이터 분석의 흐름 (오늘 수업 순서)
# | 단계 | 내용 | 주로 쓰는 도구 |
# |---|---|---|
# | 1. 환경 준비 | 라이브러리 불러오기, 한글 폰트 설정 | `import` |
# | 2. 데이터 불러오기 & 살펴보기 | 데이터의 크기, 모양, 자료형 확인 | `read_csv`, `head`, `info`, `describe` |
# | 3. 결측치 처리 | 비어 있는 값 찾아서 채우기/버리기 | `isnull`, `fillna`, `drop` |
# | 4. 단일 변수 분석 | 성별·등급별 생존율 | `value_counts`, `groupby` |
# | 5. 다변수 분석 | 두 가지 조건을 함께 비교 | `pivot_table`, `crosstab` |
# | 6. 파생 변수 만들기 | 새로운 열(가족 수, 나이대, 호칭) 만들기 | `np.where`, `pd.cut`, `str.extract` |
# | 7. 시각화 | 그래프로 한눈에 보기 | `matplotlib`, `seaborn` |
# | 8. 인사이트 & 간단한 예측 | 분석 결과로 생존 예측 규칙 만들기 | 넘파이 비교 연산 |


# %% [markdown]
# ---
# ## 1. 환경 준비
#
# 분석에 필요한 도구(라이브러리)를 불러옵니다.
# - **numpy** : 숫자 계산, 배열 연산
# - **pandas** : 표(DataFrame) 형태의 데이터 다루기
# - **matplotlib** : 그래프 그리기의 기본 도구
# - **seaborn** : matplotlib을 더 쉽고 예쁘게 쓰게 해주는 도구


# %%
# ──────────────────────────────────────────────
# 1-1. 라이브러리 불러오기
# ──────────────────────────────────────────────
import os                         # 운영체제(폴더 경로 등) 관련 기능
import platform                   # 지금 쓰는 컴퓨터가 윈도우인지 맥인지 알아낼 때 사용

import numpy as np                # 넘파이는 관례적으로 np 라는 별명으로 불러옵니다.
import pandas as pd               # 판다스는 관례적으로 pd 라는 별명으로 불러옵니다.
import matplotlib.pyplot as plt   # 그래프 도구는 plt 라는 별명으로 불러옵니다.
import seaborn as sns             # 시본은 sns 라는 별명으로 불러옵니다.

# 설치된 버전을 확인해 두면, 나중에 오류가 났을 때 원인을 찾기 쉽습니다.
print('numpy  버전 :', np.__version__)
print('pandas 버전 :', pd.__version__)


# %%
# ──────────────────────────────────────────────
# 1-2. 그래프 한글 폰트 설정
# ──────────────────────────────────────────────
# 그래프의 기본 글꼴은 한글을 지원하지 않아서 한글이 □□□ 처럼 깨집니다.
# 그래서 운영체제(OS)에 맞는 한글 글꼴을 지정해 줍니다.
if platform.system() == 'Windows':      # 윈도우라면
    font_name = 'Malgun Gothic'         # → 맑은 고딕
elif platform.system() == 'Darwin':     # 맥(macOS)이라면
    font_name = 'AppleGothic'           # → 애플고딕
else:                                   # 그 외(리눅스, 구글 코랩 등)라면
    font_name = 'NanumGothic'           # → 나눔고딕
    # 코랩에는 한글 글꼴이 없어서, 나눔고딕이 없으면 자동으로 설치합니다. (처음 한 번, 약 10~20초)
    import matplotlib.font_manager as fm            # 글꼴 목록을 관리하는 도구
    if not any(f.name == 'NanumGothic' for f in fm.fontManager.ttflist):   # 나눔고딕이 목록에 없으면
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')     # 리눅스 명령으로 글꼴 설치
        # 방금 설치한 글꼴 파일들을 matplotlib 글꼴 목록에 등록합니다. (런타임 재시작 없이 바로 사용 가능)
        for font_file in fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum']):
            fm.fontManager.addfont(font_file)

# seaborn 기본 스타일(흰 바탕 + 격자)을 적용하면서 글꼴도 함께 지정합니다.
# ※ sns.set_theme()은 글꼴 설정을 초기화하므로, 글꼴은 반드시 여기서 같이 넣어 줍니다.
sns.set_theme(style='whitegrid', font=font_name)

# 한글 글꼴을 쓰면 마이너스(-) 기호가 깨지는 문제가 있어서 이를 막아 줍니다.
plt.rcParams['axes.unicode_minus'] = False

# 판다스 출력 옵션: 열이 많아도 생략(...) 없이 모두 보여 주기
pd.set_option('display.max_columns', None)


# %%
# ──────────────────────────────────────────────
# 1-3. 데이터 주소 지정 (깃허브에서 바로 내려받기)
# ──────────────────────────────────────────────
# 데이터 파일을 직접 내려받을 필요 없이, 깃허브에 올려 둔 CSV 파일을 '인터넷 주소(URL)'로 바로 읽습니다.
# → 내 PC, 주피터, 구글 코랩 어디서든 똑같이 실행됩니다. (인터넷 연결 필요)
#
# raw.githubusercontent.com 주소란?
#   깃허브 웹페이지(github.com/...) 주소가 아니라, 파일의 '원본 내용(raw)'만 돌려주는 주소입니다.
#   pd.read_csv() 에는 반드시 이 raw 주소를 넣어야 합니다.
#   주소 형식 : https://raw.githubusercontent.com/<계정>/<저장소>/<브랜치>/<파일이름>
DATA_URL = 'https://raw.githubusercontent.com/sinaboro/data/main/titanic_kaggle.csv'

# ⚠ 같은 저장소의 titanic.csv 는 seaborn 버전(열 이름이 소문자, Name·Cabin 열 없음)이라
#    이 예제와 열 구성이 다릅니다. 반드시 Kaggle 원본인 titanic_kaggle.csv 를 사용하세요.
#
# (참고) 인터넷이 안 되는 곳에서는 titanic.csv(Kaggle 원본)를 노트북과 같은 폴더에 두고
#        아래 줄의 # 을 지워서 사용하세요.
# DATA_URL = 'titanic.csv'

print('데이터 주소 :', DATA_URL)


# %% [markdown]
# ---
# ## 2. 데이터 불러오기 & 첫인상 살펴보기
#
# 데이터 분석의 첫 단계는 **"이 데이터가 어떻게 생겼는지"** 를 아는 것입니다.
# 처음 보는 데이터를 만나면 항상 다음을 확인하세요.
# 1. 몇 행, 몇 열인가? → `shape`
# 2. 실제 값은 어떻게 생겼나? → `head()`, `tail()`, `sample()`
# 3. 각 열의 자료형과 빈 값은? → `info()`
# 4. 숫자 열의 분포는? → `describe()`


# %%
# ──────────────────────────────────────────────
# 2-1. CSV 파일 불러오기
# ──────────────────────────────────────────────
# pd.read_csv() : 쉼표(,)로 구분된 텍스트 파일(CSV)을 읽어서 DataFrame(표)으로 만들어 줍니다.
#                 파일 경로 대신 인터넷 주소(URL)를 넣으면 내려받으면서 바로 읽어 줍니다.
df = pd.read_csv(DATA_URL)

# 열 이름이 예제와 같은지 확인합니다. (다른 버전의 타이타닉 파일을 읽으면 아래에서 KeyError 가 납니다)
expected_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
missing_cols = [c for c in expected_cols if c not in df.columns]   # 없는 열만 골라냄
if missing_cols:
    print('⚠ 이 예제와 다른 형식의 파일입니다. 없는 열 :', missing_cols)
    print('  → Kaggle 원본 타이타닉 파일(titanic_kaggle.csv)을 사용하세요.')

# 원본 데이터는 따로 복사해 두면, 실수로 데이터를 망가뜨려도 언제든 다시 시작할 수 있습니다.
df_raw = df.copy()

# shape : (행 개수, 열 개수)를 튜플로 알려 줍니다.
print('데이터 크기(행, 열) :', df.shape)
print(f'→ 승객 {df.shape[0]}명, 정보 {df.shape[1]}가지')


# %%
# head() : 위에서부터 5줄을 보여 줍니다. 괄호 안에 숫자를 넣으면 그만큼 보여 줍니다.
print(df.head())


# %%
# tail(3) : 맨 아래 3줄을 보여 줍니다.
print(df.tail(3))


# %%
# sample(5) : 무작위로 5줄을 뽑아서 보여 줍니다.
# random_state 를 고정하면 몇 번을 실행해도 같은 행이 뽑힙니다. (결과 재현 가능)
print(df.sample(5, random_state=42))


# %% [markdown]
# ### 📋 컬럼(열) 설명
# | 컬럼 | 의미 | 값의 예 |
# |---|---|---|
# | `PassengerId` | 승객 번호 | 1, 2, 3 … |
# | `Survived` | **생존 여부 (분석 대상!)** | 0 = 사망, 1 = 생존 |
# | `Pclass` | 객실 등급 | 1 = 1등석, 2 = 2등석, 3 = 3등석 |
# | `Name` | 이름 (호칭 포함) | Braund, **Mr.** Owen Harris |
# | `Sex` | 성별 | male, female |
# | `Age` | 나이 | 22.0, 0.42(아기) … |
# | `SibSp` | 함께 탄 형제자매(Sibling) + 배우자(Spouse) 수 | 0, 1, 2 … |
# | `Parch` | 함께 탄 부모(Parent) + 자녀(Child) 수 | 0, 1, 2 … |
# | `Ticket` | 티켓 번호 | A/5 21171 |
# | `Fare` | 운임(요금) | 7.25, 71.28 … |
# | `Cabin` | 객실 번호 | C85 (빈 값이 매우 많음) |
# | `Embarked` | 탑승 항구 | C = 셰르부르, Q = 퀸스타운, S = 사우샘프턴 |


# %%
# ──────────────────────────────────────────────
# 2-2. 데이터 구조 확인 : info()
# ──────────────────────────────────────────────
# info() 에서 확인할 것
#   - RangeIndex : 전체 행 개수
#   - Non-Null Count : 비어 있지 않은 값의 개수 → 891보다 작으면 '결측치'가 있다는 뜻!
#   - Dtype : 자료형 (int64 = 정수, float64 = 실수, object/str = 문자열)
df.info()


# %%
# ──────────────────────────────────────────────
# 2-3. 숫자 열 요약 통계 : describe()
# ──────────────────────────────────────────────
# count(개수), mean(평균), std(표준편차), min(최솟값),
# 25%/50%/75%(사분위수, 50%는 중앙값), max(최댓값)를 한 번에 보여 줍니다.
# round(2) : 소수점 둘째 자리까지 반올림해서 보기 좋게 만듭니다.
print(df.describe().round(2))


# %%
# 문자열 열 요약 : exclude='number' → 숫자가 아닌 열만 요약합니다.
#   count  : 값이 있는 개수
#   unique : 서로 다른 값의 종류 수
#   top    : 가장 많이 나온 값
#   freq   : 가장 많이 나온 값의 등장 횟수
print(df.describe(exclude='number'))


# %% [markdown]
# ### 🤔 생각해보기
# 1. `describe()` 결과에서 `Survived`의 **mean(평균)이 0.38** 입니다. 이 숫자는 무엇을 의미할까요?
#    > 힌트: 0과 1로만 이루어진 열의 평균 = 1의 비율
# 2. `Fare`(운임)의 최댓값과 중앙값(50%)을 비교해 보세요. 무엇을 알 수 있나요?
# 3. `Age`의 count가 891이 아닌 이유는?


# %% [markdown]
# ---
# ## 3. 결측치(빈 값) 처리
#
# 현실의 데이터에는 **빈 값(결측치, NaN)** 이 자주 있습니다.
# 결측치를 그대로 두면 평균 계산, 그래프, 머신러닝 등에서 문제가 생길 수 있어서
# 분석 전에 **찾아내고 → 이유를 생각하고 → 적절히 처리**해야 합니다.
#
# | 처리 방법 | 언제 쓰나? | 코드 |
# |---|---|---|
# | 삭제 | 빈 값이 너무 많거나, 아주 조금일 때 | `dropna()`, `drop()` |
# | 대표값으로 채우기 | 숫자 → 평균/중앙값, 문자 → 최빈값 | `fillna()` |
# | 그룹별 대표값으로 채우기 | 그룹마다 값의 차이가 클 때 | `groupby().transform()` |


# %%
# ──────────────────────────────────────────────
# 3-1. 열별 결측치 개수와 비율 확인
# ──────────────────────────────────────────────
# isnull() : 빈 값이면 True, 아니면 False 인 표를 만듭니다.
# .sum()   : True 는 1, False 는 0 으로 계산되므로 더하면 '빈 값의 개수'가 됩니다.
missing_count = df.isnull().sum()

# .mean() 을 하면 '빈 값의 비율'이 됩니다. (True 의 개수 ÷ 전체 개수)
missing_ratio = df.isnull().mean() * 100

# 두 결과를 옆으로 붙여서 하나의 표로 만듭니다.
missing = pd.DataFrame({'결측치 수': missing_count,
                        '결측치 비율(%)': missing_ratio.round(1)})

# 결측치가 있는 열만 골라서(불리언 인덱싱), 많은 순서로 정렬합니다.
print(missing[missing['결측치 수'] > 0].sort_values('결측치 수', ascending=False))


# %% [markdown]
# **결과 해석**
# - `Cabin` : 약 77%가 비어 있음 → 값을 채우기엔 너무 많이 비어 있음 → **"객실 정보가 있는가?"만 남기고 열 삭제**
# - `Age` : 약 20%가 비어 있음 → 중요한 정보이므로 **그룹별 중앙값으로 채우기**
# - `Embarked` : 단 2개 → **가장 많이 나온 값(최빈값)으로 채우기**


# %%
# ──────────────────────────────────────────────
# 3-2. Embarked(탑승 항구) : 최빈값으로 채우기
# ──────────────────────────────────────────────
# 빈 값이 있는 행만 확인해 봅니다.
print(df[df['Embarked'].isnull()][['PassengerId', 'Name', 'Embarked']])

# mode() : 최빈값(가장 많이 나온 값)을 구합니다.
#          최빈값이 여러 개일 수도 있어서 결과가 Series 로 나오므로 [0] 으로 첫 번째 값을 꺼냅니다.
most_common_port = df['Embarked'].mode()[0]
print('가장 많이 탄 항구 :', most_common_port)

# fillna(값) : 빈 값을 지정한 값으로 채운 '새로운 Series'를 돌려줍니다.
# ⚠ 반드시 결과를 다시 원래 열에 대입해야 실제로 바뀝니다!
df['Embarked'] = df['Embarked'].fillna(most_common_port)
print('채운 후 결측치 수 :', df['Embarked'].isnull().sum())


# %%
# ──────────────────────────────────────────────
# 3-3. Cabin(객실 번호) : 파생 변수 만들고 삭제
# ──────────────────────────────────────────────
# 객실 번호 자체보다 "객실 번호가 기록되어 있는가?" 가 더 의미 있을 수 있습니다.
# (객실 번호가 기록된 승객은 주로 1등석 승객이었습니다.)
# notnull() : 값이 있으면 True → astype(int) 로 True=1, False=0 으로 바꿉니다.
df['HasCabin'] = df['Cabin'].notnull().astype(int)

# drop(columns=[...]) : 지정한 열을 삭제한 새 DataFrame 을 돌려줍니다.
df = df.drop(columns=['Cabin'])

# HasCabin 별 생존율을 미리 살짝 확인해 볼까요?
print(df.groupby('HasCabin')['Survived'].mean().round(3))


# %%
# ──────────────────────────────────────────────
# 3-4. Age(나이) : 그룹별 중앙값으로 채우기
# ──────────────────────────────────────────────
# 방법 ① 전체 중앙값 하나로 채우기 (간단하지만 정확도가 떨어짐)
print('전체 나이 중앙값 :', df['Age'].median())

# 방법 ② 객실 등급 + 성별 그룹마다 중앙값이 다르다면, 그룹별로 채우는 게 더 정확합니다.
# 먼저 그룹별 중앙값이 정말 다른지 확인해 봅니다.
#   unstack() : 안쪽 인덱스(Sex)를 열로 펼쳐서 표 형태로 보여 줍니다.
age_median_table = df.groupby(['Pclass', 'Sex'])['Age'].median().unstack()
print(age_median_table)


# %%
# → 1등석 남성은 40세, 3등석 남성은 25세처럼 그룹마다 꽤 차이가 납니다!
#
# transform('median') : 그룹별 중앙값을 계산한 뒤,
#                       '원래 행 개수(891개) 그대로' 각 행에 해당 그룹의 중앙값을 붙여 줍니다.
#                       → 그래서 fillna() 에 바로 넣을 수 있습니다.
group_median_age = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')

# 결측치가 있는 행만 그룹별 중앙값으로 채워집니다. (원래 값이 있는 행은 그대로!)
df['Age'] = df['Age'].fillna(group_median_age)


# %%
# 모든 결측치가 처리되었는지 최종 확인합니다.
print('남은 결측치 합계 :', df.isnull().sum().sum())   # 0 이 나오면 성공!
print('처리 후 데이터 크기 :', df.shape)


# %% [markdown]
# ### 🤔 생각해보기
# 1. `Cabin`의 결측치를 최빈값으로 채운다면 어떤 문제가 생길까요?
# 2. `Age`를 **평균**이 아니라 **중앙값**으로 채운 이유는 무엇일까요?


# %% [markdown]
# ---
# ## 4. 단일 변수 분석 — "무엇이 생존과 관련 있을까?"
#
# 이제 본격적으로 질문을 던지고 데이터로 답을 찾아봅니다.
# 핵심 아이디어 : **`Survived`는 0/1 이므로 평균을 내면 곧 '생존율'** 입니다.
#
# ```python
# df.groupby('기준열')['Survived'].mean()   # 기준열 그룹별 생존율
# ```


# %%
# ──────────────────────────────────────────────
# 4-1. 전체 생존 현황
# ──────────────────────────────────────────────
# value_counts() : 값별 개수를 셉니다.
print('[생존자 수]')
print(df['Survived'].value_counts())

# normalize=True : 개수 대신 비율로 보여 줍니다.
print('\n[생존 비율]')
print(df['Survived'].value_counts(normalize=True).round(3))

# 0/1 열의 평균 = 1(생존)의 비율 → 전체 생존율
overall_rate = df['Survived'].mean()
print(f'\n전체 생존율 : {overall_rate:.1%}')     # :.1% → 백분율로 소수 첫째 자리까지 표시


# %%
# ──────────────────────────────────────────────
# 4-2. 성별에 따른 생존율
# ──────────────────────────────────────────────
# groupby('Sex') : 성별로 그룹을 나눕니다.
# ['Survived'].agg([...]) : 그룹마다 여러 통계를 한 번에 계산합니다.
#    count = 승객 수, sum = 생존자 수(1의 합), mean = 생존율
sex_survival = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])

# 열 이름을 한글로 바꿔서 보기 좋게 만듭니다.
sex_survival.columns = ['승객 수', '생존자 수', '생존율']
sex_survival['생존율'] = sex_survival['생존율'].round(3)
print(sex_survival)


# %% [markdown]
# 👉 **여성의 생존율(약 74%)이 남성(약 19%)보다 4배 가까이 높습니다.**
# "여성과 어린이 먼저(Women and children first)" 라는 구조 원칙이 실제로 지켜졌음을 데이터로 확인할 수 있습니다.


# %%
# ──────────────────────────────────────────────
# 4-3. 객실 등급에 따른 생존율
# ──────────────────────────────────────────────
pclass_survival = df.groupby('Pclass')['Survived'].agg(['count', 'mean'])
pclass_survival.columns = ['승객 수', '생존율']
print(pclass_survival.round(3))


# %%
# ──────────────────────────────────────────────
# 4-4. 탑승 항구에 따른 생존율
# ──────────────────────────────────────────────
# sort_values(ascending=False) : 생존율이 높은 순서로 정렬
print(df.groupby('Embarked')['Survived'].mean().sort_values(ascending=False).round(3))


# %%
# ──────────────────────────────────────────────
# 4-5. 생존자 vs 사망자 : 평균 비교
# ──────────────────────────────────────────────
# 이번에는 반대로, 생존 여부로 그룹을 나눠서 다른 숫자 열들의 평균을 비교해 봅니다.
print(df.groupby('Survived')[['Age', 'Fare', 'SibSp', 'Parch', 'HasCabin']].mean().round(2))


# %% [markdown]
# ### 🤔 생각해보기
# 1. 1등석 생존율은 3등석의 몇 배인가요? 왜 이런 차이가 났을까요?
# 2. 셰르부르(C) 항구 승객의 생존율이 높은 이유는 **항구 자체** 때문일까요? 다른 이유가 있을까요?
#    > 힌트: 항구별 객실 등급 구성을 확인해 보세요. → 5장에서 확인!


# %% [markdown]
# ---
# ## 5. 다변수 분석 — 두 가지 조건을 함께 보기
#
# 성별도 중요하고, 등급도 중요했습니다. 그렇다면 **"3등석 여성"** 과 **"1등석 남성"** 중 누가 더 많이 살아남았을까요?
# 두 개 이상의 조건을 함께 비교할 때는 `pivot_table` 과 `crosstab` 을 사용합니다.


# %%
# ──────────────────────────────────────────────
# 5-1. groupby 에 열 두 개 넣기
# ──────────────────────────────────────────────
# 결과는 '멀티 인덱스(인덱스가 2단)' 형태의 Series 가 됩니다.
print(df.groupby(['Sex', 'Pclass'])['Survived'].mean().round(3))


# %%
# ──────────────────────────────────────────────
# 5-2. pivot_table : 엑셀 피벗 테이블처럼 표로 정리
# ──────────────────────────────────────────────
# index   : 행(세로)으로 놓을 기준
# columns : 열(가로)로 놓을 기준
# values  : 계산할 값
# aggfunc : 계산 방법 (mean = 평균 → 생존율)
pivot = pd.pivot_table(df,
                       index='Sex',
                       columns='Pclass',
                       values='Survived',
                       aggfunc='mean')
print(pivot.round(3))


# %% [markdown]
# 👉 **1·2등석 여성은 90% 이상 생존**, 반면 **2·3등석 남성은 15% 안팎**에 그쳤습니다.
# 3등석 여성(50%)이 1등석 남성(약 37%)보다 생존율이 높습니다 → **성별의 영향이 등급보다 더 컸다**고 볼 수 있습니다.


# %%
# ──────────────────────────────────────────────
# 5-3. crosstab : 두 범주의 '개수(빈도)' 교차표
# ──────────────────────────────────────────────
# margins=True : 행/열 합계(All)를 함께 보여 줍니다.
print(pd.crosstab(df['Embarked'], df['Pclass'], margins=True))


# %%
# normalize='index' : 각 행(항구)의 합이 1이 되도록 비율로 바꿉니다.
# → 항구별로 1·2·3등석 승객이 각각 몇 % 였는지 알 수 있습니다.
print(pd.crosstab(df['Embarked'], df['Pclass'], normalize='index').round(3))


# %% [markdown]
# 👉 C 항구 승객은 **절반 이상(약 51%)이 1등석**이었습니다.
# 즉 C 항구의 생존율이 높았던 것은 항구 때문이라기보다 **1등석 승객이 많았기 때문**이라고 해석하는 것이 합리적입니다.
#
# > 📌 **상관관계 ≠ 인과관계** : 두 값이 함께 움직인다고 해서 하나가 다른 하나의 원인인 것은 아닙니다.


# %% [markdown]
# ---
# ## 6. 파생 변수 만들기 — 새로운 관점 더하기
#
# 기존 열을 조합하거나 가공해서 **분석에 더 유용한 새 열**을 만드는 것을 **파생 변수(Feature Engineering)** 라고 합니다.
#
# | 새 열 | 만드는 방법 | 사용할 도구 |
# |---|---|---|
# | `FamilySize` | 형제자매·배우자 + 부모·자녀 + 본인 | 열끼리 덧셈 |
# | `IsAlone` | 가족 수가 1명이면 1, 아니면 0 | `np.where` |
# | `AgeGroup` | 나이를 구간으로 나누기 | `pd.cut` |
# | `FareBand` | 요금을 같은 인원수로 4등분 | `pd.qcut` |
# | `Title` | 이름에서 호칭(Mr, Mrs…) 뽑기 | `str.extract` |


# %%
# ──────────────────────────────────────────────
# 6-1. FamilySize(가족 수), IsAlone(혼자 탑승 여부)
# ──────────────────────────────────────────────
# 판다스는 열끼리 바로 더할 수 있습니다. (넘파이 배열의 벡터 연산과 같은 원리!)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1    # +1 은 '나 자신'

# np.where(조건, 참일 때 값, 거짓일 때 값) : 조건에 따라 값을 골라 넣습니다.
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)

# 만든 열을 확인해 봅니다.
print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head())


# %%
# 가족 수별 승객 수와 생존율
family_survival = df.groupby('FamilySize')['Survived'].agg(['count', 'mean']).round(3)
family_survival.columns = ['승객 수', '생존율']
print(family_survival)


# %%
# 혼자 탄 사람 vs 가족과 함께 탄 사람
print(df.groupby('IsAlone')['Survived'].mean().round(3))


# %% [markdown]
# 👉 **혼자 탄 승객(30%)** 보다 **2~4인 가족(55~72%)** 의 생존율이 높았지만,
# **5인 이상 대가족**은 오히려 생존율이 크게 떨어졌습니다. (서로를 챙기느라 탈출이 늦었을 수도 있겠죠?)


# %%
# ──────────────────────────────────────────────
# 6-2. AgeGroup(나이대) : pd.cut 으로 구간 나누기
# ──────────────────────────────────────────────
# bins   : 구간의 경계값 → (0,12], (12,19], (19,39], (39,59], (59,100]
#          '(' 는 미포함, ']' 는 포함이라는 뜻입니다. 즉 12세는 '어린이'에 들어갑니다.
# labels : 각 구간에 붙일 이름 (구간 개수 = bins 개수 - 1)
df['AgeGroup'] = pd.cut(df['Age'],
                        bins=[0, 12, 19, 39, 59, 100],
                        labels=['어린이', '청소년', '청년', '중년', '노년'])

# observed=True : 실제로 존재하는 그룹만 계산하라는 옵션 (범주형 데이터에서 경고를 막아 줌)
print(df.groupby('AgeGroup', observed=True)['Survived'].agg(['count', 'mean']).round(3))


# %%
# ──────────────────────────────────────────────
# 6-3. FareBand(요금 등급) : pd.qcut 으로 인원수 기준 4등분
# ──────────────────────────────────────────────
# pd.cut  : 값의 '범위'를 기준으로 나눔 (구간마다 사람 수가 다를 수 있음)
# pd.qcut : 사람 '수'가 같아지도록 나눔 (4 → 4등분, 각 구간에 약 25%씩)
df['FareBand'] = pd.qcut(df['Fare'], 4, labels=['저가', '중저가', '중고가', '고가'])

print(df.groupby('FareBand', observed=True)['Survived'].agg(['count', 'mean']).round(3))


# %%
# ──────────────────────────────────────────────
# 6-4. Title(호칭) : 이름에서 문자열 뽑아내기
# ──────────────────────────────────────────────
# 이름 형식 : "성, 호칭. 이름"  예) "Braund, Mr. Owen Harris"
# str.extract(정규표현식) : 패턴에 맞는 부분 중 괄호( ) 안의 내용을 뽑아냅니다.
#   r',\s*([^\.]+)\.'  의 의미
#     ,      → 쉼표 다음부터
#     \s*    → 공백이 0개 이상 있고
#     ([^\.]+) → 마침표(.)가 아닌 글자들이 1개 이상 이어진 부분을 '뽑아라'  ← 이게 호칭!
#     \.     → 그리고 마침표로 끝나는 곳까지
# expand=False : 결과를 DataFrame 이 아닌 Series 로 받기
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)

# 어떤 호칭들이 있는지 개수를 세어 봅니다.
print(df['Title'].value_counts())


# %%
# 같은 뜻의 프랑스식 호칭을 영어 호칭으로 통일합니다.
#   Mlle(마드무아젤) → Miss, Ms → Miss, Mme(마담) → Mrs
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# 주요 호칭 4개를 제외한 나머지(Dr, Rev, Col 등 소수)는 'Rare'(희귀)로 묶습니다.
#   isin(리스트) : 값이 리스트 안에 있으면 True
main_titles = ['Mr', 'Miss', 'Mrs', 'Master']
df['Title'] = np.where(df['Title'].isin(main_titles), df['Title'], 'Rare')

# 호칭별 승객 수, 평균 나이, 생존율
print(df.groupby('Title').agg(승객수=('Survived', 'count'),
                        평균나이=('Age', 'mean'),
                        생존율=('Survived', 'mean')).round(2))


# %% [markdown]
# 👉 **Master** 는 어린 남자아이를 부르는 호칭입니다. (평균 나이 약 7세)
# 같은 남성이라도 **Mr(약 16%)** 와 달리 **Master(약 57%)** 는 생존율이 훨씬 높습니다 → **"어린이 먼저"** 원칙의 증거!
#
# > 📌 `agg(새열이름=('대상열', '함수'))` 형식을 쓰면 결과 열 이름을 원하는 대로 지정할 수 있습니다. (Named Aggregation)


# %%
# 지금까지 만든 파생 변수를 포함한 데이터를 확인합니다.
print(df.head())


# %% [markdown]
# ---
# ## 7. 시각화 — 그래프로 한눈에 보기
#
# 숫자 표는 정확하지만, **그래프는 차이를 한눈에** 보여 줍니다.
#
# | 그래프 | 언제 쓰나? | seaborn 함수 |
# |---|---|---|
# | 막대(개수) | 범주별 개수 비교 | `countplot` |
# | 막대(평균) | 범주별 평균(생존율) 비교 | `barplot` |
# | 히스토그램 | 숫자 데이터의 분포 | `histplot` |
# | 상자 그림 | 그룹별 분포·이상치 비교 | `boxplot` |
# | 히트맵 | 여러 변수의 상관관계 | `heatmap` |


# %%
# ──────────────────────────────────────────────
# 7-1. 생존자 수 / 성별 생존율 / 등급별 생존율 (한 번에 3개 그래프)
# ──────────────────────────────────────────────
# plt.subplots(1, 3) : 1행 3열, 즉 그래프 3개를 나란히 그릴 판(fig)과 칸(axes)을 만듭니다.
# figsize=(가로, 세로) : 그림 크기(인치)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ① 생존자/사망자 수 : countplot 은 각 값의 '개수'를 막대로 그립니다.
sns.countplot(data=df, x='Survived', ax=axes[0])     # ax=axes[0] : 첫 번째 칸에 그리기
axes[0].set_title('생존자 수 (0=사망, 1=생존)')
axes[0].set_xlabel('생존 여부')
axes[0].set_ylabel('승객 수')

# ② 성별 생존율 : barplot 은 y값의 '평균'을 막대로 그립니다. → Survived 평균 = 생존율
#    errorbar=None : 막대 위의 오차 막대(검은 선)를 표시하지 않음
sns.barplot(data=df, x='Sex', y='Survived', errorbar=None, ax=axes[1])
axes[1].set_title('성별 생존율')
axes[1].set_ylabel('생존율')

# ③ 등급별 + 성별 생존율 : hue='Sex' 를 주면 성별로 색을 나눠서 막대를 그립니다.
sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', errorbar=None, ax=axes[2])
axes[2].set_title('객실 등급·성별 생존율')
axes[2].set_xlabel('객실 등급')
axes[2].set_ylabel('생존율')

plt.tight_layout()   # 그래프끼리 겹치지 않게 간격을 자동 조절
plt.show()           # 그래프를 화면에 표시


# %%
# ──────────────────────────────────────────────
# 7-2. 나이 분포 : 생존자 vs 사망자 히스토그램
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 4.5))

# histplot : 숫자 데이터를 구간(bins)으로 나눠 각 구간의 개수를 막대로 그립니다.
#   hue='Survived'     : 생존 여부에 따라 색을 나눔
#   multiple='stack'   : 두 그룹의 막대를 위로 쌓아서 표시
#   bins=30            : 구간을 30개로 나눔
# ※ 결측치를 채우기 전의 원본(df_raw) 나이를 사용해야 '중앙값 쏠림' 없이 실제 분포를 볼 수 있습니다.
sns.histplot(data=df_raw, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('나이 분포 (생존 여부별)')
plt.xlabel('나이')
plt.ylabel('승객 수')
plt.show()


# %% [markdown]
# 👉 **10세 미만 어린이 구간**에서는 생존자(주황) 비율이 높고, **20~30대 구간**에서는 사망자(파랑)가 압도적으로 많습니다.


# %%
# ──────────────────────────────────────────────
# 7-3. 등급별 요금 분포 : 상자 그림(boxplot)
# ──────────────────────────────────────────────
plt.figure(figsize=(8, 4.5))

# boxplot 읽는 법
#   상자 가운데 선 = 중앙값, 상자의 위/아래 = 75%/25% 지점,
#   수염 밖의 점 = 이상치(다른 값들과 동떨어진 값)
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title('객실 등급별 요금 분포')
plt.xlabel('객실 등급')
plt.ylabel('요금')
plt.ylim(0, 300)      # 512 같은 극단값 때문에 상자가 납작해지지 않도록 y축 범위를 제한
plt.show()


# %%
# ──────────────────────────────────────────────
# 7-4. 가족 수별 생존율 : 꺾은선 그래프(pointplot)
# ──────────────────────────────────────────────
plt.figure(figsize=(8, 4.5))
sns.pointplot(data=df, x='FamilySize', y='Survived', errorbar=None)
plt.axhline(overall_rate, color='gray', linestyle='--', label='전체 평균 생존율')  # 기준선
plt.title('가족 수별 생존율')
plt.xlabel('가족 수 (본인 포함)')
plt.ylabel('생존율')
plt.legend()
plt.show()


# %%
# ──────────────────────────────────────────────
# 7-5. 상관관계 히트맵
# ──────────────────────────────────────────────
# 상관계수(corr) : -1 ~ 1 사이의 값
#   +1 에 가까울수록 : 한쪽이 커질 때 다른 쪽도 커짐
#   -1 에 가까울수록 : 한쪽이 커질 때 다른 쪽은 작아짐
#    0 에 가까울수록 : 별 관계 없음
# 상관계수는 숫자로만 계산되므로, 성별을 숫자(남=0, 여=1)로 바꾼 열을 하나 만듭니다.
df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1})   # map(사전) : 값을 사전에 따라 바꿈

corr_cols = ['Survived', 'Pclass', 'Sex_num', 'Age', 'Fare', 'FamilySize', 'HasCabin']
corr = df[corr_cols].corr()     # 상관계수 표 계산

plt.figure(figsize=(8, 6))
# annot=True : 칸 안에 숫자 표시 / fmt='.2f' : 소수 둘째 자리 / cmap : 색상 (파랑-흰색-빨강)
# vmin, vmax : 색의 기준 범위를 -1 ~ 1 로 고정
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('변수 간 상관관계')
plt.show()

# 생존(Survived)과의 상관계수만 큰 순서대로 정리
print(corr['Survived'].drop('Survived').sort_values(ascending=False).round(2))


# %% [markdown]
# 👉 생존과 가장 관련이 큰 변수는 **성별(+0.54)**, 그다음은 **객실 등급(-0.34, 숫자가 작을수록=높은 등급일수록 생존)**, **객실 정보 유무(+0.32)**, **요금(+0.26)** 순입니다.


# %% [markdown]
# ---
# ## 8. 인사이트 정리 & 간단한 생존 예측
#
# ### 📝 지금까지 발견한 사실
# 1. **성별** : 여성 생존율(74%)이 남성(19%)보다 압도적으로 높다. → 가장 강력한 요인
# 2. **객실 등급** : 1등석(63%) > 2등석(47%) > 3등석(24%). 높은 등급일수록 생존율이 높다.
# 3. **나이** : 어린이(특히 Master 호칭의 남자아이)는 생존율이 높았다.
# 4. **가족 수** : 2~4인 가족이 가장 생존율이 높고, 혼자이거나 5인 이상이면 낮았다.
# 5. **탑승 항구** : C 항구 생존율이 높았지만, 이는 1등석 승객 비율이 높았기 때문이다.
#
# ### 🔮 분석 결과로 생존을 예측해 보자!
# 머신러닝 없이, **우리가 찾은 규칙**만으로 생존 여부를 맞혀 봅니다.
# 정확도 = 예측이 맞은 승객 수 ÷ 전체 승객 수


# %%
# ──────────────────────────────────────────────
# 8-1. 기준선 : "모두 사망했다"고 예측하면?
# ──────────────────────────────────────────────
# np.zeros(n) : 0 이 n 개 들어 있는 넘파이 배열 → 모두 0(사망)으로 예측
pred_all_dead = np.zeros(len(df), dtype=int)

# (예측 == 실제) : 맞으면 True, 틀리면 False 인 배열 → mean() 하면 맞힌 비율(정확도)
acc_base = (pred_all_dead == df['Survived'].values).mean()
print(f'규칙 0 (모두 사망)          정확도 : {acc_base:.1%}')


# %%
# ──────────────────────────────────────────────
# 8-2. 규칙 1 : "여성은 생존, 남성은 사망"
# ──────────────────────────────────────────────
pred_sex = np.where(df['Sex'] == 'female', 1, 0)
acc_sex = (pred_sex == df['Survived'].values).mean()
print(f'규칙 1 (여성=생존)          정확도 : {acc_sex:.1%}')


# %%
# ──────────────────────────────────────────────
# 8-3. 규칙 2 : 규칙 1 + 우리가 찾은 예외 반영
# ──────────────────────────────────────────────
#  - 여성이지만 3등석이면서 대가족(5명 이상)이면 → 사망
#  - 남성이지만 남자아이(Master)이고 1·2등석이면 → 생존
# & : 그리고(AND), | : 또는(OR), ~ : 아니다(NOT)  ※ 조건마다 괄호( )를 꼭 쳐야 합니다!
is_female      = df['Sex'] == 'female'
is_poor_bigfam = (df['Pclass'] == 3) & (df['FamilySize'] >= 5)
is_rich_boy    = (df['Title'] == 'Master') & (df['Pclass'] <= 2)

survive_rule = (is_female & ~is_poor_bigfam) | is_rich_boy
pred_rule2 = np.where(survive_rule, 1, 0)

acc_rule2 = (pred_rule2 == df['Survived'].values).mean()
print(f'규칙 2 (여성 + 예외 반영)   정확도 : {acc_rule2:.1%}')


# %%
# 세 규칙의 정확도를 막대그래프로 비교합니다.
result = pd.Series({'모두 사망': acc_base,
                    '여성=생존': acc_sex,
                    '여성+예외': acc_rule2})

plt.figure(figsize=(7, 4))
ax = sns.barplot(x=result.index, y=result.values)
ax.bar_label(ax.containers[0], fmt='%.3f')   # 막대 위에 값 표시
plt.title('생존 예측 규칙별 정확도')
plt.ylabel('정확도')
plt.ylim(0, 1)
plt.show()


# %% [markdown]
# 👉 단순한 규칙만으로도 **약 82%** 의 정확도를 얻었습니다!
# 이렇게 **데이터에서 패턴을 찾아 규칙을 만드는 과정**을 컴퓨터가 자동으로 하도록 만든 것이 바로 **머신러닝**입니다.
# (다음 단계: `scikit-learn` 의 결정 트리, 로지스틱 회귀 등)


# %% [markdown]
# ---
# ## 9. 오늘 배운 내용 정리
#
# | 하고 싶은 일 | 판다스/넘파이 코드 |
# |---|---|
# | CSV 불러오기 | `pd.read_csv('파일.csv')` |
# | 크기·구조 보기 | `df.shape`, `df.info()`, `df.head()` |
# | 요약 통계 | `df.describe()` |
# | 결측치 개수 | `df.isnull().sum()` |
# | 결측치 채우기 | `df['열'] = df['열'].fillna(값)` |
# | 그룹별 채우기 | `df.groupby('그룹')['열'].transform('median')` |
# | 열 삭제 | `df.drop(columns=['열'])` |
# | 값별 개수 | `df['열'].value_counts()` |
# | 그룹별 생존율 | `df.groupby('열')['Survived'].mean()` |
# | 교차 분석 | `pd.pivot_table(...)`, `pd.crosstab(...)` |
# | 조건으로 값 만들기 | `np.where(조건, 참값, 거짓값)` |
# | 구간 나누기 | `pd.cut(...)`, `pd.qcut(...)` |
# | 문자열 추출 | `df['열'].str.extract(정규식)` |
# | 그래프 | `sns.countplot`, `sns.barplot`, `sns.histplot`, `sns.boxplot`, `sns.heatmap` |
#
# ### ✏️ 다음 할 일
# `04_실습문제_타이타닉` 파일의 문제를 풀어 보세요. 오늘 배운 코드를 그대로 응용하면 풀 수 있습니다!


# %%
# (선택) 전처리가 끝난 데이터를 파일로 저장해 두면 다음 수업(머신러닝)에서 바로 사용할 수 있습니다.
# index=False : 0, 1, 2 … 행 번호는 저장하지 않음
# encoding='utf-8-sig' : 엑셀에서 열어도 한글이 깨지지 않도록 하는 인코딩
df.to_csv('titanic_clean.csv', index=False, encoding='utf-8-sig')
print('저장 완료 : titanic_clean.csv')
