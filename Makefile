do:
	nohup uvicorn main:app --port 8888 --host 0.0.0.0 --reload /dev/null 2>&1 &
