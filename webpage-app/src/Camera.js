import React, { useCallback } from 'react';
import Webcam from "react-webcam";
import axios from 'axios';

var url = "http://127.0.0.1:5000";
var val = 0;

export default class WebcamCapture extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0,
            start: false,
            intervalId: null
        };
    }
    
    setRef = webcam => {
        this.webcam = webcam;
    };
      
    capture = () => {
        const screenshot = this.webcam.getScreenshot();
        this.setState({ screenshot })
    };

    timer = () => {
        if(this.start){
            this.intervalId = setInterval(() => {
                this.capture();
                val = makePostRequest(url + "/api/image", this.state.screenshot, onSuccess);
                console.log(val);
                //displayValue(this.val);
            }, 2000);
        }else{
            clearInterval(this.intervalId);
        }
    }

    startOn = () => {
        this.start = true;
        this.timer();
    }

    startOff = () => {
        this.start = false;
        this.timer();
    }
      
    render() {
    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
        };
      
    return (
        <div>
            <Webcam id="cam"
                audio={false}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                />
            <button onClick={this.startOn}>Start</button>
            <button onClick={this.startOff}>Stop</button>
            {this.state.screenshot ? <img src ={this.state.screenshot} /> : null}

        </div>
        );
    }
}

 function makePostRequest(url, data, onSuccess) {
    var xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if(xhr.readyState == 4) {
            onSuccess(xhr.response);
        }
    }
    xhr.open('POST', url, false);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8')
    xhr.send(JSON.stringify({"data": data}))
}

var displayValue = function(data) {
    //TODO: Show in text box
    console.log(data);
}

var onSuccess = function(data) {
    console.log(data)
}

var onFailure = function() {
    console.error("Error in POST");
}
