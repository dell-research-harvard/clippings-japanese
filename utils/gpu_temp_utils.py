###Function to get temperature of each gpu and change the fan speed accordingly
import subprocess
import sys
import time
import wandb


def get_gpu_temp():
    """Get GPU temperature"""
    try:
        gpu_temp = subprocess.check_output(
            "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader",
            shell=True,
            universal_newlines=True,
        )
        gpu_temp = gpu_temp.splitlines()
        gpu_temp = [int(i) for i in gpu_temp]
        return gpu_temp
    except subprocess.CalledProcessError:
        print("Error getting GPU temperature")
        sys.exit(1)

###Function that changes the fan speed
###If the temperture is above 80 degrees, the fan speed will be set to 100%
###If the temperature is below 70 degrees, the fan speed will be set to 80%
###Then, it will be be set to default speed

def check_default_setting(string,i):
    """Parse the default setting"""
    ##Check if the default setting is 1 or 0
    string=string.strip()
    print(string)
    if string == "Attribute 'GPUFanControlState' ([gpu:" + str(i) + "]): 1.":
        return 1
    elif string == "Attribute 'GPUFanControlState' ([gpu:" + str(i) + "]): 0.":
        return 0
    else:
        print("Error parsing the default setting")
        sys.exit(1)
   


def set_fan_speed(gpu_temp):
    """Set fan speed"""
    fan_speed_pre = get_fan_speed()
    ##Iterate over all the gpus
    for i in range(0, len(gpu_temp)):
        if gpu_temp[i] > 80:
            if fan_speed_pre[i] == 100:
                print("GPU " + str(i) + " is hot, fan speed is already at 100% - no change")
                continue
            print("GPU " + str(i) + " is hot, setting fan speed to 100%")
            subprocess.call(
                "sudo nvidia-settings -a [gpu:" + str(i) + "]/GPUFanControlState=1 -a [fan:" + str(i) + "]/GPUTargetFanSpeed=100 --display=:0",
                shell=True
            )

        elif gpu_temp[i] <= 80 and gpu_temp[i] > 70:
            if fan_speed_pre[i] == 90:
                print("GPU " + str(i) + " is warm, fan speed is already at 90% - no change")
                continue
            print("GPU " + str(i) + " is warm, setting fan speed to 90%")
            subprocess.call(
                "sudo nvidia-settings -a [gpu:" + str(i) + "]/GPUFanControlState=1 -a [fan:" + str(i) + "]/GPUTargetFanSpeed=90 --display=:0",
                shell=True
            )
        else:
            ##Check if fan already at default speed from nvidia-setings
            default_setting=subprocess.check_output("sudo nvidia-settings -query [gpu:" + str(i) + "]/GPUFanControlState --display=:0",
             shell=True,universal_newlines=True)
            default_setting=default_setting.splitlines()[1]
            print(default_setting)
            default_setting=check_default_setting(default_setting,i)
            print(default_setting)
            if default_setting == 0:
                print("GPU " + str(i) + " is cool, fan speed is already at default - no change")
                continue

            print("GPU " + str(i) + " is cool, setting fan speed to default")
            subprocess.call(
                "sudo nvidia-settings -a [gpu:" + str(i) + "]/GPUFanControlState=0 --display=:0",
                shell=True
            )


##Get fan speed
def get_fan_speed():
    """Get fan speed"""
    try:
        fan_speed = subprocess.check_output(
            "nvidia-smi --query-gpu=fan.speed --format=csv,noheader",
            shell=True,
            universal_newlines=True,
        )
        fan_speed = fan_speed.splitlines()
        ###Remove the % sign, trim the string and convert to int
        fan_speed = [i.replace("%", "") for i in fan_speed]
        fan_speed = [int(i) for i in fan_speed]
        return fan_speed
    except subprocess.CalledProcessError:
        print("Error getting fan speed")
        sys.exit(1)

###Main function
###It has to iterate over all the gpus
###It will get the temperature of each gpu and change the fan speed accordingly
###IT will also take a x seconds break before checking the temperature again
def manage_temp(sleep_sec=10,wandb_log=False):
    """Main function"""
    print("Ensuring that the fan speed is set according to the gpu temperature...")    

    gpu_temp=get_gpu_temp()
    set_fan_speed(gpu_temp)
    ##Sleep for x seconds
    print("Leting the gpu cool down for " + str(sleep_sec) + " seconds")
    time.sleep(sleep_sec)
    ##Log fan speed to wandb
    fan_speed_list = get_fan_speed()        
    if wandb_log:
        ##Log the new fan speed
        for i in range(0, len(gpu_temp)):
            wandb.log({"train/GPU_" + str(i) + "_fanspeed": fan_speed_list[i]})


def reset_fan_speed_all():
    """Reset fan speed to default"""
    print("Resetting fan speed to default")

    gpu_temp=get_gpu_temp()
    for i in range(0, len(gpu_temp)):
        subprocess.call(
            "sudo nvidia-settings -a [gpu:" + str(i) + "]/GPUFanControlState=0 --display=:0",
            shell=True
        )


if __name__ == "__main__":

    ###Run the function
    manage_temp(10)

    ###Run the function every 5 minutes
    while True:
        manage_temp(240)

    ###Reset the fan speed to default
    reset_fan_speed_all()
        



    




