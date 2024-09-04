
from flask import Flask, request, jsonify
from threading import Thread
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter, RequestLimit
import helper_candidate_faqs_country_cases

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[])


vector_dbs = helper_candidate_faqs_country_cases.load_all_country_vectordbs()
print("vectordb loaded for all location")

#returning custome messsage when request limit is exceeded.
def index_ratelimit_error_responder(request_limit: RequestLimit):
    return jsonify({"error": "rate_limit_exceeded, please try again later."})



@app.before_request
def authenticate():
    print("Inside Authenticate")
    print(request.headers.get('X-API-KEY'))
    api_key = request.headers.get('X-API-KEY')
    if not helper_candidate_faqs_country_cases.check_api_key(api_key):
        return jsonify({'message': 'Authentication failed'}), 401


    
@app.route('/', methods=['GET'])
def main_route():
    return "Genpact Candidate Bot Dev API V1.0 App is working"


@app.route('/recruitement_assist/dev/faq/country/getaianswer', methods=['POST'])
#@limiter.limit("1000 per minute", on_breach=index_ratelimit_error_responder)
def get_answer_candidate_country_faqs():
    try:
        question = request.json['question']
        location = request.json['location']
        print(f"User Query : {question}")

        # Generate a random job ID
        job_id = helper_candidate_faqs_country_cases.generate_job_id()
        print("Job ID:", job_id)

        # Check if the location-specific vector database exists
        if location in vector_dbs.keys():
            location_vector_db = vector_dbs[location]
            print(f"{location} vector db fetched")
        else:
            location_vector_db = vector_dbs["candidate_bot_corpus_V14_27june2024"]
            print(f"candidate_bot_corpus_V14_27june2024 vector db loaded")

        Thread(target=helper_candidate_faqs_country_cases.get_answer_task_country, args=(question, location_vector_db, job_id)).start()

        responsemsg = f"Answer is being fetched for Job ID:  {job_id}"
        response = {"response": responsemsg,
                    "Job_Id": job_id}
        return response

    except Exception as e:
        return jsonify({"error": f"An error occurred in get ai answer: {str(e)}"}), 500

@app.route('/recruitement_assist/dev/faq/country/fetchaianswer', methods=['POST'])
#@limiter.limit("1000 per minute",on_breach=index_ratelimit_error_responder)
def fetch_answer_candidate_country_faqs():
    try:
        Job_Id = request.json['Job_Id']
        print(f"User Job Id : {Job_Id}")

        response = helper_candidate_faqs_country_cases.read_job_answer(Job_Id)
        answer = response["Answer"]
        sources = response["Sources"]
        response = {"answer": answer,
                    "sources": sources,
                    "Job_Id": Job_Id}
        return response
    except Exception as e:
        return jsonify({"error": f"An error occurred in fetch ai answer: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=8080)