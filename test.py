

class test:
    count = 0
    def __init__(self, string):
        print(string, test.count)
        test.count += 1



if __name__ == "__main__":
    a = test("a")
    b = test("b")