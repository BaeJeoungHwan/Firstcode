import pandas as pd
import matplotlib.pyplot as plt
# num_series = pd.Series([1,2,3,4,5], index = ["index_0,","index_1","index_2","index_3","index_4"])
# num_dataframe = pd.DataFrame([1,2,3,4,5],index = ["index_0,","index_1","index_2","index_3","index_4"])

# print(num_series)
# print(num_dataframe)

titanic_df = pd.read_csv("lab04_titanic/train.csv")

# print(titanic_df.info())

alive = titanic_df[titanic_df["Survived"] == 1]
dead = titanic_df[titanic_df["Survived"] == 0]


# print(len(alive), "/", len(titanic_df))
# print(len(dead), "/", len(titanic_df))

# plt.bar(["alive","dead"], height = [len(alive), len(dead)])
# plt.show()

#scatter 이용해서 요금별 탑승자 시각화하기
# plt.scatter(titanic_df["PassengerId"], titanic_df["Fare"])
# plt.xlabel("PassengerID")
# plt.ylabel("Fare")
# plt.show()

#각각을 색상분류 후 시각화 하기
# plt.scatter(alive["PassengerId"], alive["Fare"], color="GREEN")
# plt.scatter(dead["PassengerId"], dead["Fare"], color="RED")
# plt.xlabel("PassengerID")
# plt.ylabel("Fare")
# plt.show()

#50 달러 기준 탑승자 수 구해보기
Under_50dlr = titanic_df[titanic_df["Fare"] < 50]
Over_50dlr = titanic_df[titanic_df["Fare"] >= 50]

# print("$50 미만 요금 탑승자 수 : ",len(Under_50dlr),"(",len(Under_50dlr)/len(titanic_df)*100,")")
# print("$50 이상 요금 탑승자 수 : ",len(Over_50dlr))

# 생존자를 $50 달러 기준으로 나눠보기
alive_over_50 = Over_50dlr[Over_50dlr["Survived"] == 1]
alive_under_50 = Under_50dlr[Under_50dlr["Survived"] == 1]

plt.subplot(2,1,1)
plt.xlabel("$50 미만 생존 비율")
plt.pie([len(alive_under_50), len(Under_50dlr)-len(alive_under_50)], colors = ["GREEN","RED"], labels = ["Alive","Dead"])
plt.subplot(2,1,2)
plt.xlabel("$50 이상 생존 비율")
plt.pie([len(alive_over_50), len(Over_50dlr)-len(alive_over_50)], colors = ["GREEN","RED"], labels = ["Alive","Dead"])
plt.show()