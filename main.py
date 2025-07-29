import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
import neattext as nt
import neattext.functions as nfx

# 1. УЛУЧШЕННАЯ ПРЕДОБРАБОТКА ТЕКСТА
def advanced_text_preprocessing(text):
    """Расширенная предобработка текста"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = nfx.remove_stopwords(text, lang='ru')
    
    return text.strip()

def create_marker_words_features(df, target_cols):
    """Создание признаков на основе слов-маркеров для каждого класса"""
    
    marker_words = {
        'ЗП': ['зарплат', 'оклад', 'деньги', 'выплат', 'доход', 'мало', 'недостаточно', 'финанс', 'бюджет'],
        'График': ['график', 'время', 'смен', 'переработк', 'выходн', 'час', 'режим', 'ночн', 'сверхурочн'],
        'Взаимоотношения с руководителем / коллегами': ['начальник', 'руководител', 'коллег', 'отношен', 'конфликт', 'команд', 'общен'],
        'Условия труда': ['условия', 'рабочее место', 'оборудован', 'комфорт', 'температур', 'освещен', 'безопасност'],
        'Отсутствие карьерного роста': ['карьер', 'рост', 'развит', 'повышен', 'перспектив', 'будущ', 'продвижен'],
        'Стресс': ['стресс', 'нервн', 'напряжен', 'устал', 'выгоран', 'психолог', 'давлен', 'беспокойств'],
        'Переезд': ['переезд', 'перееха', 'смен', 'местожительств', 'другой город', 'семь', 'жилье'],
        'Транспортная доступность': ['дорог', 'транспорт', 'добират', 'далеко', 'метро', 'автобус', 'пробк'],
        'Функционал': ['обязанност', 'задач', 'функци', 'работ', 'делать', 'должност', 'функционал'],
        'Социальный пакет': ['соц', 'льгот', 'страховк', 'отпуск', 'больничн', 'компенсац', 'бонус'],
        'Проблемы с адаптацией': ['адаптац', 'привык', 'новичок', 'ориентац', 'влива', 'знаком'],
        'Состояние здоровья сотрудника / родственника': ['здоров', 'болезн', 'лечен', 'врач', 'больниц', 'родственник'],
        'Ушел на другую работу': ['другая работ', 'новое место', 'предложен', 'лучш', 'интересн'],
        'Нет возможности совмещать с учебой': ['учеб', 'институт', 'университет', 'студент', 'экзамен', 'сессия'],
        'Призыв на воинскую службу / уход на СВО': ['армия', 'военн', 'призыв', 'служб', 'сво', 'мобилизац'],
        'Заканчиваются разрешительные документы': ['документ', 'виза', 'разрешен', 'патент', 'регистрац', 'закончился']
    }
    
    df = df.copy()
    
    # Создание признаков для каждой категории
    for category, words in marker_words.items():
        if category in target_cols:
            # Подсчет количества маркерных слов
            df[f'{category}_marker_count'] = df['Комментарий_processed'].apply(
                lambda x: sum(1 for word in words if word in x.lower())
            )
            
            # Булевый признак наличия маркерных слов
            df[f'{category}_has_markers'] = (df[f'{category}_marker_count'] > 0).astype(int)
    
    return df

def create_syntactic_features(df):
    """Создание синтаксических признаков"""
    df = df.copy()
    
    df['sentence_count'] = df['Комментарий'].str.count('[.!?]+')
    
    df['avg_sentence_length'] = df['text_length'] / (df['sentence_count'] + 1)
    
    df['comma_count'] = df['Комментарий'].str.count(',')
    
    df['comma_density'] = df['comma_count'] / (df['word_count'] + 1)
    
    conjunctions = ['и', 'а', 'но', 'или', 'либо', 'также', 'тоже', 'либо', 'ни', 'да']
    df['conjunction_count'] = df['Комментарий_processed'].apply(
        lambda x: sum(1 for conj in conjunctions if f' {conj} ' in f' {x} ')
    )
    
    prepositions = ['в', 'на', 'с', 'по', 'за', 'к', 'от', 'для', 'до', 'при', 'под', 'над', 'через']
    df['preposition_count'] = df['Комментарий_processed'].apply(
        lambda x: sum(1 for prep in prepositions if f' {prep} ' in f' {x} ')
    )
    
    df['preposition_density'] = df['preposition_count'] / (df['word_count'] + 1)

    negations = ['не', 'нет', 'ни', 'без', 'отсутств']
    df['negation_count'] = df['Комментарий_processed'].apply(
        lambda x: sum(1 for neg in negations if neg in x.lower())
    )
    
    positive_words = ['хорош', 'отличн', 'замечательн', 'прекрасн', 'нравит', 'понравил', 'доволен']
    df['positive_words_count'] = df['Комментарий_processed'].apply(
        lambda x: sum(1 for word in positive_words if word in x.lower())
    )
    
    negative_words = ['плох', 'ужасн', 'отвратительн', 'не нравит', 'недоволен', 'проблем', 'трудност']
    df['negative_words_count'] = df['Комментарий_processed'].apply(
        lambda x: sum(1 for word in negative_words if word in x.lower())
    )
    
    intensifiers = ['очень', 'крайне', 'чрезвычайно', 'совершенно', 'абсолютно', 'максимально']
    df['intensifier_count'] = df['Комментарий_processed'].apply(
        lambda x: sum(1 for word in intensifiers if word in x.lower())
    )
    
    return df

def create_text_features(df, target_cols):
    """Создание всех дополнительных текстовых признаков"""
    df = df.copy()
    
    df['text_length'] = df['Комментарий'].str.len()
    df['word_count'] = df['Комментарий'].str.split().str.len()
    df['exclamation_count'] = df['Комментарий'].str.count('!')
    df['question_count'] = df['Комментарий'].str.count('\?')
    df['capital_count'] = df['Комментарий'].str.count('[А-ЯЁ]')
    
    df = create_marker_words_features(df, target_cols)
    
    df = create_syntactic_features(df)
    
    return df

# 3. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
def improved_multilabel_classification():
    df = pd.read_excel('training_data.xlsx')

    all_cat = set()
    for cat in df['Категория']:
        catgrs = [c.strip() for c in cat.split(';')]
        all_cat.update(catgrs)
    all_cat = sorted(list(all_cat))
    
    for cat in all_cat:
        df[cat] = df['Категория'].apply(lambda x: 1.0 if cat in [c.strip() for c in x.split(';')] else 0.0)
    
    df['Комментарий'] = df['Комментарий'].fillna('').astype(str)
    df['Комментарий_processed'] = df['Комментарий'].apply(advanced_text_preprocessing)
    
    df = create_text_features(df, target_cols)
    
    target_cols = ['Взаимоотношения с руководителем / коллегами', 'График', 'ЗП', 
                   'Заканчиваются разрешительные документы', 'Нет возможности совмещать с учебой', 
                   'Отсутствие карьерного роста', 'Переезд', 'Призыв на воинскую службу / уход на СВО', 
                   'Проблемы с адаптацией', 'Состояние здоровья сотрудника / родственника', 
                   'Социальный пакет', 'Стресс', 'Транспортная доступность', 'Условия труда', 
                   'Ушел на другую работу', 'Функционал']
    
    y = df[target_cols]
    
    # 4. УЛУЧШЕННАЯ ВЕКТОРИЗАЦИЯ
    # TF-IDF с оптимизированными параметрами
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2'
    )

    X_text = tfidf.fit_transform(df['Комментарий_processed']).toarray()
    
    numeric_cols = [col for col in df.columns if col not in ['Комментарий', 'Комментарий_processed', 'Категория'] + target_cols]
    X_numeric = df[numeric_cols].fillna(0).values
    
    print(f"Количество текстовых признаков: {X_text.shape[1]}")
    print(f"Количество числовых признаков: {X_numeric.shape[1]}")
    print(f"Общее количество признаков: {X_text.shape[1] + X_numeric.shape[1]}")
    
    X = np.hstack([X_text, X_numeric])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y.iloc[:, 0])
    
    # 5. ОБУЧЕНИЕ РАЗЛИЧНЫХ МОДЕЛЕЙ
    models_to_test = {
        'RandomForest + BinaryRelevance': BinaryRelevance(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        'LogisticRegression + BinaryRelevance': BinaryRelevance(LogisticRegression(max_iter=1000, random_state=42)),
        'RandomForest + ClassifierChain': ClassifierChain(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        'LogisticRegression + ClassifierChain': ClassifierChain(LogisticRegression(max_iter=1000, random_state=42)),
        'GradientBoosting + BinaryRelevance': BinaryRelevance(GradientBoostingClassifier(n_estimators=100, random_state=42)),
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        print(f"Обучение модели: {name}")
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        hamming = hamming_loss(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        f1_micro = f1_score(y_test, predictions, average='micro')
        
        results[name] = {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-micro: {f1_micro:.4f}\n")
    
    return results, X_train, X_test, y_train, y_test, models_to_test

# 6. ДОПОЛНИТЕЛЬНЫЕ ТЕХНИКИ УЛУЧШЕНИЯ
def ensemble_approach(X_train, X_test, y_train, y_test):
    """Ансамблевый подход для дальнейшего улучшения"""
    
    model1 = BinaryRelevance(RandomForestClassifier(n_estimators=200, random_state=42))
    model2 = ClassifierChain(LogisticRegression(max_iter=1000, random_state=42))
    model3 = BinaryRelevance(GradientBoostingClassifier(n_estimators=100, random_state=42))
    
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    
    pred1 = model1.predict_proba(X_test)
    pred2 = model2.predict_proba(X_test)
    pred3 = model3.predict_proba(X_test)
    
    ensemble_pred_proba = (pred1 + pred2 + pred3) / 3
    ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, ensemble_pred)
    hamming = hamming_loss(y_test, ensemble_pred)
    f1_macro = f1_score(y_test, ensemble_pred, average='macro')
    
    print("Ensemble Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1-macro: {f1_macro:.4f}")
    
    return ensemble_pred

# 7. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ
def hyperparameter_tuning(X_train, y_train):
    """Настройка гиперпараметров для лучшей модели"""
    
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 15, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    model = BinaryRelevance(RandomForestClassifier(random_state=42))
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    
    sample_size = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    grid_search.fit(X_train[indices], y_train.iloc[indices])
    
    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший скор:", grid_search.best_score_)
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    results, X_train, X_test, y_train, y_test, models = improved_multilabel_classification()
    
    # Дополнительные техники
    print("\n" + "="*50)
    print("ENSEMBLE APPROACH:")
    ensemble_pred = ensemble_approach(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING:")
    best_model = hyperparameter_tuning(X_train, y_train)
