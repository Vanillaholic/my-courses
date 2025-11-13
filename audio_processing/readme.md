此处存放深蓝学院《语音信号处理》的课程资料

课程链接：https://www.shenlanxueyuan.com/course/800/task/33900/show

代码使用说明：代码文件是c语言编写的相关程序，使用时需要cmake

- add_reverb:给音频文件audio.raw添加混响，通过overlap-add算法实现与RIR的卷积
- LMS_RLS_filter:实现了LMS和RLS滤波，x(n)和d(n)都是audio.raw，滤波器系数最后会收敛成 $ \delta(n)$
