import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 데이터 처리를 위해 pandas 활용 추천

def visualize_results():
    # 1. 데이터 로드
    rewards_data = np.load("results/Combined_ppo_rewards.npy", allow_pickle=True)
    logs_data = np.load("results/infer_logs.npy", allow_pickle=True)
    
    # 2. Logs 데이터를 DataFrame으로 변환 (리스트의 딕셔너리 구조 처리)
    # pandas를 쓰면 복잡한 키(key)별 데이터 추출을 한 줄로 끝낼 수 있습니다.
    logs_df = pd.DataFrame(list(logs_data))
    
    # 3. 그래프 그리기 (서브플롯 활용)
    # figsize를 충분히 키워 겹치지 않게 합니다.
    fig = plt.figure(figsize=(12, 10))
    
    # 상단: 전체 학습 Reward (1차원 데이터)
    plt.subplot(2, 1, 1)
    plt.plot(rewards_data, label='Training Rewards')
    plt.title("PPO Training Rewards History")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 하단: Inference Logs (DataFrame 활용)
    # 여기서는 예시로 'reward'와 'speed' 두 개를 겹쳐서 보여줍니다.
    # 필요에 따라 subplot을 더 나누거나 다른 컬럼('B', 'tau')을 추가하세요.
    plt.subplot(2, 1, 2)
    plt.plot(logs_df['i'], logs_df['reward'], label='Step Reward', marker='o')
    plt.plot(logs_df['i'], logs_df['speed'], label='Speed', marker='x')
    plt.title("Inference Logs Analysis")
    plt.xlabel("Step (i)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("visualization_result.png")
    print("그래프가 저장되었습니다.")

if __name__ == "__main__":
    visualize_results()