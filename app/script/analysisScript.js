
let categories = ['positive', 'negative', 'neutral'];
let analysisLog;

async function getData(){
   let data = sessionStorage.getItem("classification");
   analysisLog = JSON.parse(data);
}

async function getLogByCategory(logs, category){

   let logList = [];
   for (let item of logs){
      if(item.Emotion == category){
         logList.push(item);
      }
   }

   return logList;
}

async function showAnalysis(){
   await getData();

   let tweetList = analysisLog.classification;
   
   analysisContainer = document.getElementById("analysisContainer");

   for (let category of categories){
      let logList = await getLogByCategory(tweetList, category);
      // If there are logs for the current category
      if(logList.length != 0){
         let iconClass = ""
         if(category == "positive"){
            iconClass = "fa-solid fa-face-smile-beam"
         } else if(category == "negative"){
            iconClass = "fa-solid fa-face-frown"
         } else {
            iconClass = "fa-solid fa-face-meh"
         }

         let header = 
            '<div class="row p-2 titleRow">' +
               '<div class="col-12"><i class="'+iconClass+'" style="color: white;"></i> ' + category.toUpperCase()  + ' TWEETS: <span style="font-weight: 700; color: #f17300;">'+logList.length+'</span></div>' +
           '</div>';
         analysisContainer.innerHTML += header
         tweets = ""
         for (let item of logList) {
            newTweet = '<div class="row tweetContainer">' + 
                    '<div class="col-2">' +
                        '<i class="fa-solid fa-circle-user" style="color: white; font-size: 1.5em"></i> <br>' + item.Username +
                    '</div>' +
                    '<div class="col-6">' + 
                        item.Text +
                    '</div>' +
                    '<div class="col-3">' +
                        item.Views + '<span style="font-weight: 700;">Visualizzazioni </span>' + 
                    '</div>' +
                    '<div class="col-1">' +
                        '<a target="_blank" href="' + item.Link + '"><span style="font-weight: 700;">Link: </span></a>' +
                    '</div>' +
                '</div>'
            tweets += newTweet
         }

         tweets = '<div class="row folderSection" style="height: 50vh; overflow-y: scroll; overflow-x: hidden;">' + tweets + '</div>'
         analysisContainer.innerHTML += tweets
      }
   }
}