import datetime


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M')
