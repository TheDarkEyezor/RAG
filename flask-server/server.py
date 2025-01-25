# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from agents import receptionist_prompt_response, Lawyer

app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['GET', 'POST'])
def members():
    if request.method == 'POST':
        data = request.get_json()
        agent = data.get('agent')
        message = data.get('message')

        if agent == 'receptionist':
            print(f"Received as receptionist: {message}")
            response = receptionist_prompt_response(message)
            
            lawyer = Lawyer(response["lawyer"], message)
            app.lawyer = lawyer
            
            # Get first question from lawyer
            first_question = lawyer.ask_question()
            
            return jsonify({
                'agent': "lawyer",  # Change to lawyer to trigger lawyer UI
                'message': f"I'm your {response['lawyer']} specialist. {first_question}"
            })
        
        elif agent == 'lawyer':
            print(f"Received as lawyer: {message}")
            if not hasattr(app, 'lawyer'):
                return jsonify({'error': 'Session expired'})
                
            response = app.lawyer.answer_question(message)
            
            # Check if response is a dict (meaning secretary report)
            if isinstance(response, dict):
                return jsonify({
                    'agent': "secretary",
                    'message': response['message'],
                    'report': response['report']
                })
            
            return jsonify({
                'agent': "lawyer",
                'message': response
            })
    
    return jsonify({
        'agent': 'receptionist',
        'message': 'How can I help you today?'
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)