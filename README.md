# swarm-proyect

docker swarm init --advertise-addr 192.168.0.28

docker network create -d overlay frontend_ntw 
docker network create -d overlay backend_ntw

docker service create --name voteapp -p 5000:80 --network frontend_ntw --replicas 5 eddlihuisi/daea_movie_ratings-vote

docker service create --name redis --replicas 5 --network frontend_ntw redis:alpine

docker service create --name worker --network frontend_ntw --network backend_ntw eddlihuisi/daea_movie_ratings-worker

docker volume create db-data

docker service create --name db --network backend_ntw -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres --mount type=volume,source=db-data,target=/var/lib/postgresql/data postgres:alpine

docker service create --name resultapp -p 5001:80 --network backend_ntw --replicas 5 eddlihuisi/daea_movie_ratings-result

