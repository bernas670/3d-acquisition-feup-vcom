rmmod v4l2loopback_dc
insmod /lib/modules/`uname -r`/kernel/drivers/media/video/v4l2loopback-dc.ko width=$1 height=$2
echo "options v4l2loopback_dc width=$1 height=$2" > /etc/modprobe.d/droidcam.conf