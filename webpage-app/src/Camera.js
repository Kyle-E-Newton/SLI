import React from 'react';
import Webcam from "react-webcam";
import axios from 'axios';

var url = "127.0.0.1:3000";
var val = 0;

export default class WebcamCapture extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0,
            start: true
        };
    }
    
    setRef = webcam => {
        this.webcam = webcam;
    };
      
    capture = () => {
        const screenshot = this.webcam.getScreenshot();
        this.setState({ screenshot })
    };

    timer = setInterval((start) => {
        this.capture();
        if (this.state.screenshot != null) {
            val = makePostRequest(url + "/api/image", this.state.screenshot);
            displayValue(this.val);
        }
    }, 2000);


      
    render() {
    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
        };
      
    return (
        <div>
            <Webcam 
                audio={false}
                height={350}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                width={350}
                videoConstraints={videoConstraints}
                />
            <button onClick={this.capture}>Capture photo</button>
            {this.state.screenshot ? <img src ={this.state.screenshot} /> : null}
        </div>
        );
    }
}

async function makePostRequest(url, data) {
    const response = await axios.post(
        url,
        {data}
    );
    return response;
}

var displayValue = function(data) {
    //TODO: Show in text box
    console.log(data);
}

var onFailure = function() {
    console.error("Error in POST");
}
