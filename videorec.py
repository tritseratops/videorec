import cv2, datetime, pandas, re, os, sys

class VideoRecorder:
    def __init__(self):
        # set up dir
        self.rec_dir_name = "rec_" + re.sub('[^a-zA-Z0-9 \n\.]', '', str(datetime.datetime.now()))
        self.motion_start = None
        self.motion_log = []
        self.motion_frames = []

    def __create_recording_dir(self):
        try:
            os.mkdir(self.rec_dir_name)
        except OSError:
            print("Creation of the directory %s failed" % self.rec_dir_name)
        else:
            print("Successfully created the directory %s " % self.rec_dir_name)
            
        return self.rec_dir_name

    def __write_motion_episode(self, start, frames):
        # filename generation
        rec_file_name = self.rec_dir_name+"/rec_" + re.sub('[^a-zA-Z0-9 \n\.]', '', str(start))+".avi"

        print(rec_file_name)
        print(frames[0])
        height, width, layers = frames[0].shape
        size = (width, height)

        out = cv2.VideoWriter(rec_file_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(frames)):
            out.write(frames[i])
        out.release()

    def start_recording(self):
        # set up recording
        self.__setup_recording()

        while self.video_in_progress:

            motion_detected = False

            #

            self.video_in_progress, frame = self.video.read()
            gray_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
            gray_frame = cv2.GaussianBlur(src=gray_frame, ksize=(21, 21), sigmaX=0)
            if self.background_frame is None:
                self.background_frame = gray_frame
                continue

            delta_frame = cv2.absdiff(self.background_frame, gray_frame)

            sug_thresh, threshhold_delta_frame = cv2.threshold(src=delta_frame, thresh=30, maxval=25,
                                                               type=cv2.THRESH_BINARY)

            # smooth delta
            threshhold_delta_frame = cv2.dilate(src=threshhold_delta_frame, kernel=None, iterations=2)

            contours, _ = cv2.findContours(image=threshhold_delta_frame.copy(), mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            # when motion detected
            # draw countour around moving objects
            for contour in contours:
                if cv2.contourArea(contour) < 10000:
                    continue
                # check if motion started
                if self.motion_start is None:
                    self.motion_start = datetime.datetime.now()
                self.background_same_motion_start_time = self.motion_start
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)

            # check if motion stopped
            if not motion_detected and not self.motion_start is None:
                self.motion_log, self.motion_start, self.motion_frames = self.__end_motion_episode_recording(self.motion_log, self.motion_start, self.motion_frames)


            # if motion is detected add frame to motion
            if motion_detected:
                self.motion_frames.append(frame)
                # check if motion is fake, like background changed

                # update background frame with it
                # if movement is 5sec, compare this frame with first moving frame
                sec_from_motion_start = (datetime.datetime.now()-self.background_same_motion_start_time).total_seconds()
                if(sec_from_motion_start>5):
                    # and if it is the same and all in moving frame are too
                    if (self.__are_frames_same(frame, self.motion_frames[0])):
                        motion_frames_same_5_sec = True
                        for motion_frame in self.motion_frames:
                            if not self.__are_frames_same(frame, motion_frame):
                                same_background_start_time = datetime.datetime.now()
                                motion_frames_same_5_sec = False
                                break
                        # if frames did not change during last 5 seconds, update background frame
                        if motion_frames_same_5_sec:
                            gray_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
                            gray_frame = cv2.GaussianBlur(src=gray_frame, ksize=(21, 21), sigmaX=0)
                            background_frame = gray_frame
                            self.background_same_motion_start_time = datetime.datetime.now()
                            self.motion_log, self.motion_start, self.motion_frames = self.__end_motion_episode_recording(self.motion_log, self.motion_start, self.motion_frames)




                
            # show videos
            cv2.imshow("Video", gray_frame)
            cv2.imshow("Delta", delta_frame)
            cv2.imshow("Threshhold", threshhold_delta_frame)
            cv2.imshow("Motion", frame)
            
            # check for end trigger
            key = cv2.waitKey(1)
            if key == ord('q'):
                # check for last motion
                if not self.motion_start is None:
                    self.motion_log, self.motion_start, self.motion_frames =  self.__end_motion_episode_recording(self.motion_log,
                                                                                                                  self.motion_start,
                                                                                                                  self.motion_frames)
                break
            # print(motion_detected)

        # print(motions)
        
        # save log to file
        df = pandas.DataFrame(columns=["Start", "End"])
        for motion in self.motion_log:
            print(motion)
            df = df.append(motion, ignore_index=True)
        print(df)
        df.to_csv(self.rec_dir_name+ "\motions.csv")
        
        self.video.release()
        cv2.destroyAllWindows()

    def __setup_recording(self):
        self.background_frame = None
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video_in_progress = True

        # set up dir
        self.rec_dir_name = self.__create_recording_dir()
        self.background_same_motion_start_time = None

    def __are_frames_same(self, one_frame, other_frame):
        gray_frame = cv2.cvtColor(src=one_frame, code=cv2.COLOR_RGB2GRAY)
        gray_frame = cv2.GaussianBlur(src=gray_frame, ksize=(21, 21), sigmaX=0)

        delta_frame = cv2.absdiff(self.background_frame, gray_frame)

        sug_thresh, threshhold_delta_frame = cv2.threshold(src=delta_frame, thresh=30, maxval=25,
                                                           type=cv2.THRESH_BINARY)

        # smooth delta
        threshhold_delta_frame = cv2.dilate(src=threshhold_delta_frame, kernel=None, iterations=2)

        contours, _ = cv2.findContours(image=threshhold_delta_frame.copy(), mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return True
        else:
            return False

    def __review_background_frame(self):
        pass

    def __end_motion_episode_recording(self, motion_log, motion_start, motion_frames):
        motion_end = datetime.datetime.now()
        motion_log.append({"Start": str(motion_start), "End": str(motion_end)})
        self.__write_motion_episode(motion_start, motion_frames)
        motion_start = None
        motion_frames = []
        return motion_log, motion_start, motion_frames




vr = VideoRecorder()
vr.start_recording()
sys.exit()

