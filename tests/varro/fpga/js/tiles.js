// This script highlights given tiles in the prjtrellis-db docs. https://symbiflow.github.io/prjtrellis-db/

var script = document.createElement("script");
  script.setAttribute("src", "https://ajax.googleapis.com/ajax/libs/jquery/1.6.4/jquery.min.js");
  script.addEventListener('load', function() {
    var script = document.createElement("script");
    document.body.appendChild(script);
  }, false);
  document.body.appendChild(script);

tiles = ['R2C124', 'R3C124']

tiles.forEach(element => $("em:contains("+element+")").parent().css({"border-color": "#f00", 
             "border-width":"5px", 
             "border-style":"dashed"}));
