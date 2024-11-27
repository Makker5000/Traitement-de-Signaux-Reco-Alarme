from Analysis import *
from Processing import *
from Comparison import *
from Result import *

def main():
    print("La première fonction du programme lol")
    alarm_type = runComparison(rate_test, test_alarm)
    generate_alarm_image(alarm_type)


main()