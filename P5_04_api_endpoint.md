# API endpoint

You can make a *request* using *curl* at the following address:
https://stackoverflow-questions.herokuapp.com/predict

# Disclaimer

The *API* is hosted on the free *Heroku* service, which requires some time to wake-up from sleep.

Please be patient for the *response* when sending a first *request*.
The following *requests* will be returned quicker.

# Make a request

The *curl* request must include a dictionary containing:
* 'title': the title of the question (*string*)
* 'body': the body of the question (*string* that might contain *html code*)

# Example request

    curl

    --header "Content-Type: application/json" 

    --data '{"title":"Getting perforce CL using shell script", "body":"Working on an assignment to have a duel play out amongst three players with varying accuracy and needs them to shoot in order. Aaron has an accuracy of 1/3, Bob has an accuracy of 1/2, and Charlie never misses. A duel should loop until one is left standing. Here is my code so far and it only ever causes the first two players to miss and Charlie wins even when the random number generator should constitute a hit."}' 

    --request POST https://stackoverflow-questions.herokuapp.com/predict
