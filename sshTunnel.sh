# at master node
sudo service gmetad restart
sudo service ganglia-monitor restart
sudo service apache2 restart
sudo ufw disable

start-dfs.sh
start-yarn.sh
mr-jobhistory-daemon.sh start historyserver

ssh gpu20-x1 "sudo -S su - -c 'service ganglia-monitor restart'"
ssh gpu20-x2 "sudo -S su - -c 'service ganglia-monitor restart'"

# gpu20 tunnel (need redo after restart)
ssh -Nf -L 202.45.128.135:10098:10.42.2.40:80 10.42.2.40
ssh -Nf -L 202.45.128.135:20120:10.42.2.40:50070 10.42.2.40
ssh -Nf -L 202.45.128.135:20220:10.42.2.40:8088 10.42.2.40
ssh -Nf -L 202.45.128.135:20320:10.42.2.40:19888 10.42.2.40
ssh -Nf -L 202.45.128.135:20420:10.42.2.70:8042 hduser@10.42.2.70
ssh -Nf -L 202.45.128.135:20520:10.42.2.100:8042 hduser@10.42.2.100
ssh -Nf -L 202.45.128.135:20620:10.42.2.40:18080 10.42.2.40