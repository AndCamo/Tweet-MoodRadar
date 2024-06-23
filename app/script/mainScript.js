function sayHello() {
   console.log("CIAO")
}

async function downloadTweet(){
   let query_input = document.getElementById("query-input");
   let number_input = document.getElementById("number-input");
   let select = document.getElementById("mode-select");

   let query = query_input.value;
   let mode = select.value;
   let amount = number_input.value;

   console.log(query + " " + mode);

   let n = await eel.downloadTweet(query, amount ,mode)();

   alert("FATTO!")
}


async function getFileName(){
   let fileChooser = document.getElementById('tweet-file');
   let fileName = "";

   if(fileChooser.value != ""){
      fileName = fileChooser.files[0].name;
   } else {
      fileName = "empty"
   }
   return fileName;
}

async function startTweetAnalysis(){
   let fileName = await getFileName();
   if(fileName == "empty"){
      alert("Devi prima caricare un file!")
      return;
   }
   console.log(fileName);
   let jsonResponse = await eel.startAnalysis(fileName)();
   let response = JSON.parse(jsonResponse);
   
   console.log(response);
   if(response.status == "error"){
      alert("Impossibile analizzare questo file!")
   } else {
      sessionStorage.setItem("classification", jsonResponse);
      window.location.href = 'analysisResult.html';
   }

}

