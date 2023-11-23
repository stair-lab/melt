# # This script enables the user to upload a file to Google Drive
# # fail fast
set -e pipefail
CLIENT_ID=294247525422-t5fr1iij3dto6k3usmjvm5hj0fukshhn.apps.googleusercontent.com
CLIENT_SECRET=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
SCOPE=https://www.googleapis.com/auth/drive.file # This is the URL we'll send the user to first to get their authorization
# verify device
VERIFY_DEVICE=`curl -d "client_id="$CLIENT_ID"&scope=$SCOPE" https://oauth2.googleapis.com/device/code`
echo $VERIFY_DEVICE
# extract device_code value from the json response using jq
DEVICE_CODE=`echo $VERIFY_DEVICE | python3 -c "import sys, json; print(json.load(sys.stdin)['device_code'])"`
USER_CODE=`echo $VERIFY_DEVICE | python3 -c "import sys, json; print(json.load(sys.stdin)['user_code'])"`
echo "Please access this url to add authenticated code: https://www.google.com/device"
echo "Here is your code:"
echo $USER_CODE
# pause the script to give the user time to navigate to verification_url and enter the user_code.
echo "Press any key if you have added to your device..."
read
read -p "Enter The Path of Your File: " PATH_FILE
FILE_NAME=$(basename $PATH_FILE)
echo $FILE_NAME
# get bearer code
BEARER=`curl -d client_id=$CLIENT_ID \
 -d client_secret=$CLIENT_SECRET \
 -d device_code=$DEVICE_CODE \
 -d grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code https://accounts.google.com/o/oauth2/token`
echo $BEARER
# extract access_token value from the json response using jq
ACCESS_TOKEN=`echo $BEARER | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])"` 
echo $ACCESS_TOKEN
echo `curl -X POST -L \
 -H 'Authorization: Bearer '${ACCESS_TOKEN} \
 -F 'metadata={name : "'${FILE_NAME}'", parents : ["1M0u86L-y4HIt2K7QRhnmdLUE1Q5y3eIJ"]};type=application/json;charset=UTF-8' \
 -F 'file=@'${PATH_FILE}';type=application/zip' \
 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart'`