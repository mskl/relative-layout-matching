.PHONY: build run stop bash token restart logs events tsp

build:
	docker-compose -f docker-compose.train.gpu.yml build

run: build
	docker-compose -f docker-compose.train.gpu.yml up -d --remove-orphans
	docker-compose -f docker-compose.train.yml logs -f

stop:
	docker-compose -f docker-compose.train.gpu.yml down

bash:
	docker-compose -f docker-compose.train.gpu.yml exec jupyter bash

token:
	docker-compose -f docker-compose.train.gpu.yml exec jupyter jupyter notebook list

restart:
	docker-compose -f docker-compose.train.gpu.yml restart

logs:
	docker-compose -f docker-compose.train.gpu.yml logs -f

events:
	docker events --since 24h

tsp:
	docker-compose -f docker-compose.train.gpu.yml exec jupyter tsp -t
