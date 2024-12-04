from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from sklearn.metrics import jaccard_score
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split


# 데이터가 너무 커서 pivot table의 용량을 다 사용한다고 오류가 나와서 샘플링해서 진행했습니다.
def sampling_data(processed_data):# 기본 데이터 추출
    # 사용자별 평가 횟수 계산
    rating_users = processed_data['User-ID'].value_counts().reset_index()
    rating_users.columns = ['User-ID', 'Count']
    # 책별 평가 횟수 계산
    rating_books = processed_data['ISBN'].value_counts().reset_index()
    rating_books.columns = ['ISBN', 'Count']
    # 필터링 적용
    processed_data = processed_data[processed_data['User-ID'].isin(rating_users[rating_users['Count'] > 250]['User-ID'])]
    processed_data = processed_data[processed_data['ISBN'].isin(rating_books[rating_books['Count'] > 50]['ISBN'])]
    # 중복된 데이터 제거
    processed_data.drop_duplicates(inplace=True)
    return processed_data

def model_selection(sampled_data):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(sampled_data[['User-ID', 'Title', 'Rating']], reader)
    # 알고리즘 정의
    algorithms = {
        'SVD': SVD(),
        'KNNBasic': KNNBasic(),
        'KNNWithMeans': KNNWithMeans(),
        'KNNWithZScore': KNNWithZScore()
    }
    results = {}
    # 각 알고리즘 평가
    for name, algo in algorithms.items():
        cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)          
        results[name] = {
            'RMSE': cv_results['test_rmse'].mean(),
            'MAE': cv_results['test_mae'].mean()
        }   
    # 결과 출력
    print("\n=== Final Results ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Average RMSE: {metrics['RMSE']:.4f}")
        print(f"Average MAE: {metrics['MAE']:.4f}")
    
    return results

def hyper_parameter(train_data):
    # 데이터 로더 생성 (평점 범위 설정)
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(train_data[['User-ID', 'Title', 'Rating']], reader)
    # 원시 평점 데이터를 셔플하여 훈련/테스트 데이터 분리
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)                  
    threshold = int(len(raw_ratings) * 0.8)  # 80%를 훈련 데이터로 사용
    train_raw_ratings = raw_ratings[:threshold]  # 훈련용 평점 데이터
    test_raw_ratings = raw_ratings[threshold:]   # 테스트용 평점 데이터
    # 훈련 및 테스트 셋 생성
    data.raw_ratings = train_raw_ratings
    trainset = data.build_full_trainset()  
    testset = data.construct_testset(test_raw_ratings)
    # 하이퍼파라미터 그리드 설정
    param_grid = {
        "n_factors": range(10, 100, 20),  # 잠재 요인 수 범위
        "n_epochs": [5, 10, 20],          # 학습 에포크 수
        "lr_all": [0.002, 0.005],         # 학습률
        "reg_all": [0.2, 0.5]             # 정규화 파라미터
    }
    # SVD 모델에 대한 그리드 서치 실행
    gridsearchSVD = GridSearchCV(SVD, param_grid, measures=['mae', 'rmse'], cv=5, n_jobs=-1)
    gridsearchSVD.fit(data)
    # 최적의 MAE와 RMSE 하이퍼파라미터 및 점수 출력
    print(f'MAE Best Parameters:  {gridsearchSVD.best_params["mae"]}')
    print(f'MAE Best Score:       {gridsearchSVD.best_score["mae"]}\n')
    print(f'RMSE Best Parameters: {gridsearchSVD.best_params["rmse"]}')
    print(f'RMSE Best Score:      {gridsearchSVD.best_score["rmse"]}\n')

def svd_recommendations(train_data, test_user_id, k=5):
    reader = Reader(rating_scale=(train_data['Rating'].min(), train_data['Rating'].max()))
    data = Dataset.load_from_df(train_data[['User-ID', 'Title', 'Rating']], reader)
    # 학습 및 테스트 데이터 생성
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
    # SVD 모델 학습
    algo_svd = SVD()
    algo_svd.fit(trainset) 
    # 특정 사용자를 위한 추천 생성
    test_user_books = train_data[train_data['User-ID'] == test_user_id]['Title'].values
    predictions = [
        (title, algo_svd.predict(test_user_id, title).est) 
        for title in train_data['Title'].unique() if title not in test_user_books
    ]
    # 평점이 높은 순으로 정렬하여 상위 k개의 추천 반환
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
    return [title for title, rating in recommendations]
           
# 코사인 유사도 기반 Collaborative filtering 함수
def recommend_books_cosine(user_book_matrix, user_id, top_n=5):
    user_similarity = cosine_similarity(user_book_matrix) # 입력된 user_book_matrix를 바탕으로 코사인 유사도 계산
    # 위의 결과를 바탕으로 user_id와의 sim_score를 저장한 Series 데이터를 만들고 정렬
    sim_scores = pd.Series(user_similarity[user_book_matrix.index.get_loc(user_id)], 
                           index=user_book_matrix.index)
    sim_scores = sim_scores.drop(index=user_id).sort_values(ascending=False)
    # 위의 구해진 정렬된 유사도를 통해 상위 top_n 명의 점수를 가져와서 비슷한 user를 찾음
    similar_users = sim_scores.index[:top_n]
    # user_book_matrix에서 위의 similar_users의 책을 찾아서 정렬하고 read_books에 입력된 user_id의 사용자가 읽은 책을 저장해서 drop
    recommended_books = user_book_matrix.loc[similar_users].mean().sort_values(ascending=False)
    read_books = user_book_matrix.loc[user_id][user_book_matrix.loc[user_id] > 0].index
    recommended_books = recommended_books.drop(index=read_books, errors='ignore')
    # top_n만큼 책 추천
    return list(recommended_books.index[:top_n])

# 유클리디안 거리 기반 Collaborative filtering 함수 : 로직이 크게 다르지 않아서 설명은 생략합니다. 필요시에 말씀해주시면 주석 처리할게요
def recommend_books_euclidean(user_book_matrix, user_id, top_n=5):
    distances = user_book_matrix.apply(lambda x: euclidean(user_book_matrix.loc[user_id], x), axis=1)
    distances = distances.drop(index=user_id).sort_values()
    
    similar_users = distances.index[:top_n]
    recommended_books = user_book_matrix.loc[similar_users].mean().sort_values(ascending=False)
    read_books = user_book_matrix.loc[user_id][user_book_matrix.loc[user_id] > 0].index
    recommended_books = recommended_books.drop(index=read_books, errors='ignore')
    
    return recommended_books.index[:top_n]

def content_based_filtering(author, processed_data, n_recommendations=5):
    # TF-IDF 벡터화 (Author 열 사용)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(processed_data['Author'].fillna(''))
    # 입력된 저자와 일치하는 인덱스 찾기
    author_idx = processed_data[processed_data['Author'] == author].index[0] 
    # 코사인 유사도 계산
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim_matrix[author_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]   
    # 추천 title 목록 반환
    recommended_titles = [processed_data['Title'].iloc[i[0]] for i in sim_scores]
    return recommended_titles

def content_based_knn_filtering(author, processed_data, n_recommendations=5):
    # TF-IDF로 텍스트 특징 추출 (저자 이름)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(processed_data['Author'].fillna(''))
    # KNN 모델 설정
    knn = NearestNeighbors(n_neighbors=n_recommendations+1, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    # 추천 대상 저자의 인덱스 가져오기
    author_idx = processed_data[processed_data['Author'] == author].index[0]
    # 주어진 저자와 유사한 저자 찾기
    _, indices = knn.kneighbors(tfidf_matrix[author_idx])
    # 인덱스를 바탕으로 추천 제목 리스트 생성 (자기 자신은 제외)
    recommended_titles = processed_data['Title'].iloc[indices[0][1:]].tolist()
    return recommended_titles

if __name__ == "__main__":
    pass