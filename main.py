
import WorksRetrival

START_FROM = 10


if __name__ == '__main__':
    i = START_FROM
    for num in range(START_FROM, 27779):
        try:
            i += 1
            WorksRetrival.get_work(num)
        except:
            print("Exception! Reached num ", i)

    WorksRetrival.clean_directory("./authors")

