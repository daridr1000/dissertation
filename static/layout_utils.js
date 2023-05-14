
function clearTitle(){
    document.getElementById("formTitle").value = "";
}

function clearStory(){
    document.getElementById("formStory").value = "";
}

function loading(){
    let results = document.getElementById("results");
    results.setAttribute("hidden", "hidden");
    let spinner = document.getElementById("loading");
    spinner.removeAttribute("hidden");
}

async function fetchScenario(){
  const response = await fetch('./static/scenarios.json');
  const scenarios = await response.json();
  const scenarioNumber = getRandomInt(5);
  document.getElementById("formTitle").value = scenarios[scenarioNumber]["title"];
  document.getElementById("formStory").value = scenarios[scenarioNumber]["body"];

}

function getRandomInt(index) {
  return Math.floor(Math.random() * index);
}


$('textarea').keyup(function() {
    
    var characterCount = $(this).val().length,
        current = $('#current'),
        maximum = $('#maximum'),
        theCount = $('#the-count');
      
    current.text(characterCount);

    
        
  });