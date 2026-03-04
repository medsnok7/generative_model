dsi() { docker stop $(docker ps -a | awk -v i="^$3/$1.*" '{if($2~i){print$1}}'); }
echo "removing all other containers to stop their processes"
echo "stopping $1"
dsi $1
echo "will try to pull the image $1:$2"
docker login -u $3 -p dckr_pat_76ra0RSNHTMYLBHgNBae_lZAUvQ && docker pull $3/$1:$2 && docker logout