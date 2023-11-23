# swarm-proyect

## Getting started

Download [Docker Desktop](https://www.docker.com/products/docker-desktop) for Mac or Windows. [Docker Compose](https://docs.docker.com/compose) will be automatically installed. On Linux, make sure you have the latest version of [Compose](https://docs.docker.com/compose/install/).

This solution uses Python, Node.js, .NET, with Redis for messaging and Postgres for storage.

Run in this directory to build and run the app:

```shell
docker compose up
```

The `vote` app will be running at [http://localhost:5000](http://localhost:5000), and the `results` will be at [http://localhost:5001](http://localhost:5001).

Alternately, if you want to run it on a [Docker Swarm](https://docs.docker.com/engine/swarm/), first make sure you have a swarm. If you don't, run:

```shell
docker swarm init
```

Once you have your swarm, in this directory run:

```shell
docker stack deploy --compose-file docker-stack.yml vote
```





docker swarm init --advertise-addr 192.168.0.28

docker network create -d overlay frontend_ntw 
docker network create -d overlay backend_ntw

docker service create --name voteapp -p 5000:80 --network frontend_ntw --replicas 5 eddlihuisi/daea_movie_ratings-vote

docker service create --name redis --replicas 5 --network frontend_ntw redis:alpine

docker service create --name worker --network frontend_ntw --network backend_ntw eddlihuisi/daea_movie_ratings-worker

docker volume create db-data

docker service create --name db --network backend_ntw -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres --mount type=volume,source=db-data,target=/var/lib/postgresql/data postgres:alpine

docker service create --name resultapp -p 5001:80 --network backend_ntw --replicas 5 eddlihuisi/daea_movie_ratings-result

