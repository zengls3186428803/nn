import threading


class MyThread(threading.Thread):
    def __init__(self, thread_id, spider, **kwargs):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.spider = spider
        self.kwargs = kwargs

    def run(self):
        print ("开始线程Thread" + str(self.thread_id))
        self.spider.scheduler(url=self.kwargs["url"], begin_time=self.kwargs["begin_time"], end_time=self.kwargs["end_time"], filename=self.kwargs["filename"], frequency=self.kwargs["frequency"])
        print ("退出线程Thread" + str(self.thread_id))