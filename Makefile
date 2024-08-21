do:
	docker build -t obahamondev/images .	
	docker tag obahamondev/images:latest obahamondev/images:latest
	docker login
	docker push obahamondev/images:latest
	