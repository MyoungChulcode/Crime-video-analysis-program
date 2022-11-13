
def fileopen(data):
    with open(data, 'r', encoding='UTF8') as file:
        text = file.read()

        splitdata = text.split()

    return splitdata, len(splitdata)


def count_character(data):
    count = 0

    for i in data:
        count += len(i)

    return count


data, count = fileopen("C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/features/c3d_features/Explosion/Explosion001_x264.txt")

print("공백 수 : ", count - 1)

print("공백을 제외한 문자수 : ", count_character(data))

print("공백을 포함한 문자수 : ", count_character(data) + count - 1)

print("단어 수 : ", count)


myFile=open("C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/features/c3d_features/Explosion/Explosion003_x264.txt",'r')
print(myFile.read().count("\n")+1)

myFile.close()